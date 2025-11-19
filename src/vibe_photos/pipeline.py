"""High-level M1 preprocessing pipeline orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from torch import Tensor
from utils.logging import get_logger
from vibe_photos.classifier import SceneClassifierWithAttributes, build_scene_classifier
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import open_primary_db, open_projection_db
from vibe_photos.hasher import CONTENT_HASH_ALGO, PHASH_ALGO, compute_content_hash, compute_perceptual_hash, hamming_distance_phash
from vibe_photos.ml.models import get_blip_caption_model, get_siglip_embedding_model
from vibe_photos.scanner import FileInfo, scan_roots


LOGGER = get_logger(__name__)


@dataclass
class RunJournalRecord:
    """Lightweight checkpoint for resumable pipeline execution."""

    stage: str
    cursor_image_id: Optional[str]
    updated_at: float


def load_run_journal(cache_root: Path) -> Optional[RunJournalRecord]:
    """Load the run journal from ``cache/run_journal.json`` if it exists."""

    journal_path = cache_root / "run_journal.json"
    if not journal_path.exists():
        return None

    try:
        data = json.loads(journal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.error("run_journal_load_error", extra={"path": str(journal_path), "error": str(exc)})
        return None

    stage = str(data.get("stage") or "")
    cursor = data.get("cursor_image_id")
    updated_raw = data.get("updated_at")

    try:
        updated_at = float(updated_raw)
    except (TypeError, ValueError):
        updated_at = time.time()

    if not stage:
        return None

    return RunJournalRecord(stage=stage, cursor_image_id=str(cursor) if cursor is not None else None, updated_at=updated_at)


def save_run_journal(cache_root: Path, record: RunJournalRecord) -> None:
    """Persist the run journal to ``cache/run_journal.json``."""

    cache_root.mkdir(parents=True, exist_ok=True)
    journal_path = cache_root / "run_journal.json"
    payload = {
        "stage": record.stage,
        "cursor_image_id": record.cursor_image_id,
        "updated_at": record.updated_at,
    }
    journal_path.write_text(json.dumps(payload), encoding="utf-8")


class PreprocessingPipeline:
    """Orchestrates the M1 preprocessing steps for one or more album roots."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the pipeline with application settings.

        Args:
            settings: Optional pre-loaded Settings instance. When omitted,
                configuration is loaded from ``config/settings.yaml``.
        """

        self._settings = settings or load_settings()
        self._logger = get_logger(__name__, extra={"component": "preprocess"})
        self._cache_root: Optional[Path] = None
        self._journal: Optional[RunJournalRecord] = None

    def run(self, roots: Sequence[Path], primary_db_path: Path, projection_db_path: Path) -> None:
        """Run the preprocessing pipeline for the given album roots.

        The initial implementation focuses on wiring and logging. Individual
        stages are implemented as stubs that can be expanded in dedicated
        iterations.

        Args:
            roots: Album root directories to scan.
            primary_db_path: Path to the primary operational database.
            projection_db_path: Path to the projection database for model outputs.
        """

        cache_root = projection_db_path.parent
        self._cache_root = cache_root
        self._logger.info(
            "pipeline_start",
            extra={
                "roots": [str(root) for root in roots],
                "primary_db": str(primary_db_path),
                "projection_db": str(projection_db_path),
            },
        )

        journal = load_run_journal(cache_root)
        self._journal = journal
        if journal is not None:
            self._logger.info(
                "pipeline_resume",
                extra={"stage": journal.stage, "cursor_image_id": journal.cursor_image_id},
            )

        with open_primary_db(primary_db_path) as primary_conn, open_projection_db(projection_db_path) as projection_conn:
            self._run_scan_and_hash(roots, primary_conn)
            save_run_journal(cache_root, RunJournalRecord(stage="scan_and_hash", cursor_image_id=None, updated_at=time.time()))

            self._run_perceptual_hashing_and_duplicates(primary_conn, projection_conn)
            save_run_journal(cache_root, RunJournalRecord(stage="phash_and_duplicates", cursor_image_id=None, updated_at=time.time()))

            self._run_embeddings_and_captions(primary_conn, projection_conn)
            self._run_scene_classification(primary_conn, projection_conn)

        # Best-effort: remove run journal after a successful full run to avoid
        # stale cursors influencing future runs.
        try:
            journal_path = cache_root / "run_journal.json"
            if journal_path.exists():
                journal_path.unlink()
        except OSError:
            # Non-fatal; the next run will simply see an existing journal.
            pass

        self._logger.info("pipeline_complete", extra={})

    def _run_scan_and_hash(self, roots: Sequence[Path], primary_conn) -> None:
        """Scan album roots and populate the images table with content hashes.

        This stage is responsible for maintaining the ``images`` table in the
        primary database:

        - Insert rows for new content hashes discovered under the configured roots.
        - Merge paths for existing hashes, keeping one canonical ``primary_path``.
        - Handle file content changes by moving a path from the old ``image_id``
          to the new one.
        - Update deletion status when all recorded paths for an ``image_id``
          disappear from disk.
        """

        self._logger.info("scan_and_hash_start", extra={})
        files: List[FileInfo] = list(scan_roots(roots))
        self._logger.info("scan_and_hash_files_discovered", extra={"file_count": len(files)})

        cursor = primary_conn.cursor()

        # Build a mapping from path -> image_id based on existing rows.
        path_to_image_id: Dict[str, str] = {}
        cursor.execute("SELECT image_id, all_paths FROM images")
        for row in cursor.fetchall():
            raw_paths = row["all_paths"]
            try:
                paths = json.loads(raw_paths) if raw_paths else []
            except json.JSONDecodeError:
                paths = []
            for stored_path in paths:
                path_to_image_id[str(stored_path)] = row["image_id"]

        now = time.time()

        # Scan files and upsert image records.
        for file_info in files:
            path = file_info.path.resolve()
            path_str = str(path)
            new_image_id = compute_content_hash(path)
            old_image_id = path_to_image_id.get(path_str)

            # If the file existed before but its content hash changed, detach it
            # from the previous image row.
            if old_image_id is not None and old_image_id != new_image_id:
                cursor.execute("SELECT all_paths, primary_path, status FROM images WHERE image_id = ?", (old_image_id,))
                old_row = cursor.fetchone()
                if old_row is not None:
                    try:
                        old_paths = json.loads(old_row["all_paths"]) if old_row["all_paths"] else []
                    except json.JSONDecodeError:
                        old_paths = []

                    new_paths = [p for p in old_paths if p != path_str]
                    status = old_row["status"]
                    primary_path = old_row["primary_path"]

                    if new_paths:
                        if primary_path not in new_paths:
                            primary_path = new_paths[0]
                        # If the row was previously marked deleted but still has
                        # remaining paths, restore it to active.
                        if status == "deleted":
                            status = "active"
                    else:
                        # All paths removed for this image_id; mark as deleted.
                        status = "deleted"

                    cursor.execute(
                        "UPDATE images SET all_paths = ?, primary_path = ?, status = ?, updated_at = ? WHERE image_id = ?",
                        (json.dumps(new_paths), primary_path, status, now, old_image_id),
                    )

                path_to_image_id.pop(path_str, None)
                old_image_id = None

            # Upsert the row for the new image_id.
            cursor.execute(
                "SELECT image_id, primary_path, all_paths, size_bytes, mtime, status FROM images WHERE image_id = ?",
                (new_image_id,),
            )
            existing = cursor.fetchone()

            if existing is not None:
                try:
                    paths = json.loads(existing["all_paths"]) if existing["all_paths"] else []
                except json.JSONDecodeError:
                    paths = []

                if path_str not in paths:
                    paths.append(path_str)

                primary_path = existing["primary_path"] or path_str
                status = existing["status"] or "active"
                if status == "deleted":
                    status = "active"

                cursor.execute(
                    """
                    UPDATE images
                    SET primary_path = ?, all_paths = ?, size_bytes = ?, mtime = ?, hash_algo = ?, status = ?, updated_at = ?
                    WHERE image_id = ?
                    """,
                    (
                        primary_path,
                        json.dumps(paths),
                        file_info.size_bytes,
                        file_info.mtime,
                        CONTENT_HASH_ALGO,
                        status,
                        now,
                        new_image_id,
                    ),
                )
            else:
                all_paths_json = json.dumps([path_str])
                cursor.execute(
                    """
                    INSERT INTO images (
                        image_id,
                        primary_path,
                        all_paths,
                        size_bytes,
                        mtime,
                        width,
                        height,
                        exif_datetime,
                        camera_model,
                        hash_algo,
                        phash,
                        phash_algo,
                        phash_updated_at,
                        created_at,
                        updated_at,
                        status,
                        error_message,
                        schema_version
                    ) VALUES (
                        ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, ?, ?, ?, NULL, ?
                    )
                    """,
                    (
                        new_image_id,
                        path_str,
                        all_paths_json,
                        file_info.size_bytes,
                        file_info.mtime,
                        CONTENT_HASH_ALGO,
                        now,
                        now,
                        "active",
                        1,
                    ),
                )

            path_to_image_id[path_str] = new_image_id

        primary_conn.commit()

        # Second pass: remove paths that no longer exist on disk and mark rows
        # as deleted when all paths disappear.
        now = time.time()
        cursor.execute("SELECT image_id, all_paths, primary_path, status FROM images")
        for row in cursor.fetchall():
            raw_paths = row["all_paths"]
            try:
                paths = json.loads(raw_paths) if raw_paths else []
            except json.JSONDecodeError:
                paths = []

            if not paths:
                continue

            existing_paths: List[str] = []
            for stored_path in paths:
                stored_path_str = str(stored_path)
                try:
                    if Path(stored_path_str).exists():
                        existing_paths.append(stored_path_str)
                except OSError:
                    # Treat paths that raise OS errors as missing.
                    continue

            if existing_paths == paths:
                continue

            primary_path = row["primary_path"]
            status = row["status"]

            if existing_paths:
                if primary_path not in existing_paths:
                    primary_path = existing_paths[0]
                if status == "deleted":
                    status = "active"
            else:
                status = "deleted"

            cursor.execute(
                "UPDATE images SET all_paths = ?, primary_path = ?, status = ?, updated_at = ? WHERE image_id = ?",
                (json.dumps(existing_paths), primary_path, status, now, row["image_id"]),
            )

        primary_conn.commit()
        self._logger.info("scan_and_hash_complete", extra={"file_count": len(files)})

    def _run_scene_classification(self, primary_conn, projection_conn) -> None:
        """Run lightweight scene classification based on cached embeddings."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        self._logger.info("scene_classification_start", extra={})

        cache_root = self._cache_root
        primary_cursor = primary_conn.cursor()

        classifier: SceneClassifierWithAttributes = build_scene_classifier(self._settings)
        classifier_version = classifier.classifier_version

        embedding_model_name = self._settings.models.embedding.resolved_model_name()

        # Map image_id -> existing classifier_version to detect stale rows.
        primary_cursor.execute("SELECT image_id, classifier_version FROM image_scene")
        existing_versions = {row["image_id"]: row["classifier_version"] for row in primary_cursor.fetchall()}

        # Determine which embeddings are available for the configured model.
        primary_cursor.execute(
            "SELECT image_id, embedding_path FROM image_embedding WHERE model_name = ?",
            (embedding_model_name,),
        )
        embedding_rows = primary_cursor.fetchall()
        embedding_path_by_id = {row["image_id"]: row["embedding_path"] for row in embedding_rows}

        # Only classify active images with an embedding and missing or stale classifier_version.
        primary_cursor.execute("SELECT image_id FROM images WHERE status = 'active' ORDER BY image_id")
        candidate_ids = [row["image_id"] for row in primary_cursor.fetchall()]

        target_ids: List[str] = []
        for image_id in candidate_ids:
            if image_id not in embedding_path_by_id:
                continue
            if image_id not in existing_versions:
                target_ids.append(image_id)
                continue
            if existing_versions[image_id] != classifier_version:
                target_ids.append(image_id)

        # Apply journal-based cursor if we are resuming this stage.
        if self._journal is not None and self._journal.stage == "scene_classification" and self._journal.cursor_image_id:
            cursor_id = self._journal.cursor_image_id
            target_ids = [image_id for image_id in target_ids if image_id > cursor_id]

        if not target_ids:
            self._logger.info("scene_classification_noop", extra={})
            return

        import numpy as np
        import torch

        detections_dir = cache_root / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)

        batch_size = max(1, self._settings.models.embedding.batch_size)
        updated_rows = 0

        for batch_start in range(0, len(target_ids), batch_size):
            batch_ids = target_ids[batch_start : batch_start + batch_size]
            embeddings: List[Tensor] = []
            valid_ids: List[str] = []

            for image_id in batch_ids:
                rel_path = embedding_path_by_id[image_id]
                emb_path = cache_root / "embeddings" / rel_path
                try:
                    vec = np.load(emb_path)
                except Exception as exc:
                    self._logger.error(
                        "scene_embedding_load_error",
                        extra={"image_id": image_id, "path": str(emb_path), "error": str(exc)},
                    )
                    continue

                try:
                    tensor = torch.from_numpy(vec).float()
                except Exception as exc:
                    self._logger.error(
                        "scene_embedding_tensor_error",
                        extra={"image_id": image_id, "error": str(exc)},
                    )
                    continue

                embeddings.append(tensor)
                valid_ids.append(image_id)

            if not embeddings:
                continue

            attributes_list = classifier.classify_batch(embeddings)

            now = time.time()
            for image_id, attributes in zip(valid_ids, attributes_list):
                # Persist to SQLite projection DB.
                primary_cursor.execute(
                    """
                    INSERT INTO image_scene (
                        image_id,
                        scene_type,
                        scene_confidence,
                        has_text,
                        has_person,
                        is_screenshot,
                        is_document,
                        classifier_name,
                        classifier_version,
                        updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(image_id) DO UPDATE SET
                        scene_type = excluded.scene_type,
                        scene_confidence = excluded.scene_confidence,
                        has_text = excluded.has_text,
                        has_person = excluded.has_person,
                        is_screenshot = excluded.is_screenshot,
                        is_document = excluded.is_document,
                        classifier_name = excluded.classifier_name,
                        classifier_version = excluded.classifier_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        image_id,
                        attributes.scene_type,
                        attributes.scene_confidence,
                        int(attributes.has_text),
                        int(attributes.has_person),
                        int(attributes.is_screenshot),
                        int(attributes.is_document),
                        attributes.classifier_name,
                        attributes.classifier_version,
                        now,
                    ),
                )

                # Persist a lightweight detection cache for rebuilds.
                detection_payload = {
                    "image_id": image_id,
                    "scene_type": attributes.scene_type,
                    "scene_confidence": attributes.scene_confidence,
                    "has_text": attributes.has_text,
                    "has_person": attributes.has_person,
                    "is_screenshot": attributes.is_screenshot,
                    "is_document": attributes.is_document,
                    "classifier_name": attributes.classifier_name,
                    "classifier_version": attributes.classifier_version,
                    "updated_at": now,
                }
                cache_path = detections_dir / f"{image_id}.json"
                cache_path.write_text(json.dumps(detection_payload), encoding="utf-8")
                updated_rows += 1

            # Update run journal after each batch for resumable execution.
            save_run_journal(
                cache_root,
                RunJournalRecord(stage="scene_classification", cursor_image_id=batch_ids[-1], updated_at=time.time()),
            )

        primary_conn.commit()
        self._logger.info("scene_classification_complete", extra={"updated_rows": updated_rows})

    def _run_embeddings_and_captions(self, primary_conn, projection_conn) -> None:
        """Compute SigLIP embeddings and BLIP captions for canonical images."""

        if self._cache_root is None:
            raise RuntimeError("cache_root is not initialized")

        self._logger.info("embeddings_and_captions_start", extra={})

        cache_root = self._cache_root
        primary_cursor = primary_conn.cursor()

        embedding_cfg = self._settings.models.embedding
        caption_cfg = self._settings.models.caption

        embedding_model_name = embedding_cfg.resolved_model_name()
        caption_model_name = caption_cfg.resolved_model_name()

        siglip_processor, siglip_model, siglip_device = get_siglip_embedding_model(settings=self._settings)
        blip_processor, blip_model, blip_device = get_blip_caption_model(settings=self._settings)

        # Determine active images and apply near-duplicate gating for heavy models.
        primary_cursor.execute("SELECT image_id, primary_path FROM images WHERE status = 'active' ORDER BY image_id")
        active_rows = primary_cursor.fetchall()
        paths_by_id: Dict[str, str] = {row["image_id"]: row["primary_path"] for row in active_rows}

        if not paths_by_id:
            self._logger.info("embeddings_and_captions_noop", extra={})
            return

        process_ids = sorted(paths_by_id.keys())

        if self._settings.pipeline.skip_duplicates_for_heavy_models:
            primary_cursor.execute("SELECT duplicate_image_id FROM image_near_duplicate")
            duplicate_rows = primary_cursor.fetchall()
            duplicate_ids = {row["duplicate_image_id"] for row in duplicate_rows}
            process_ids = [image_id for image_id in process_ids if image_id not in duplicate_ids]

        # Apply journal-based cursor if we are resuming this stage.
        if self._journal is not None and self._journal.stage == "embeddings_and_captions" and self._journal.cursor_image_id:
            cursor_id = self._journal.cursor_image_id
            process_ids = [image_id for image_id in process_ids if image_id > cursor_id]

        # Exclude images that already have embeddings/captions for the configured models.
        primary_cursor.execute("SELECT image_id FROM image_embedding WHERE model_name = ?", (embedding_model_name,))
        existing_embedding_ids = {row["image_id"] for row in primary_cursor.fetchall()}

        primary_cursor.execute("SELECT image_id FROM image_caption WHERE model_name = ?", (caption_model_name,))
        existing_caption_ids = {row["image_id"] for row in primary_cursor.fetchall()}

        embedding_targets = [image_id for image_id in process_ids if image_id not in existing_embedding_ids]
        caption_targets = [image_id for image_id in process_ids if image_id not in existing_caption_ids]

        embeddings_dir = cache_root / "embeddings"
        captions_dir = cache_root / "captions"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import torch

        now = time.time()

        # Embeddings
        embedding_batch_size = max(1, embedding_cfg.batch_size)
        total_embeddings = 0

        for batch_start in range(0, len(embedding_targets), embedding_batch_size):
            batch_ids = embedding_targets[batch_start : batch_start + embedding_batch_size]
            images: List["Image.Image"] = []
            valid_ids: List[str] = []

            from PIL import Image as _Image

            for image_id in batch_ids:
                path_str = paths_by_id[image_id]
                try:
                    img = _Image.open(path_str).convert("RGB")
                except Exception as exc:
                    self._logger.error(
                        "embedding_image_open_error",
                        extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                    )
                    continue

                images.append(img)
                valid_ids.append(image_id)

            if not images:
                continue

            inputs = siglip_processor(images=images, return_tensors="pt")
            inputs = inputs.to(siglip_device)

            with torch.no_grad():
                features = siglip_model.get_image_features(**inputs)

            for idx, image_id in enumerate(valid_ids):
                emb = features[idx]
                emb = emb / emb.norm(dim=-1, keepdim=True)
                vec = emb.detach().cpu().numpy().astype(np.float32)

                rel_path = f"{image_id}.npy"
                emb_path = embeddings_dir / rel_path
                np.save(emb_path, vec)

                primary_cursor.execute(
                    """
                    INSERT INTO image_embedding (
                        image_id,
                        model_name,
                        embedding_path,
                        embedding_dim,
                        model_backend,
                        updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(image_id, model_name) DO UPDATE SET
                        embedding_path = excluded.embedding_path,
                        embedding_dim = excluded.embedding_dim,
                        model_backend = excluded.model_backend,
                        updated_at = excluded.updated_at
                    """,
                    (
                        image_id,
                        embedding_model_name,
                        rel_path,
                        int(vec.shape[-1]),
                        embedding_cfg.backend,
                        now,
                    ),
                )
                total_embeddings += 1

            # Update run journal after each embedding batch.
            if batch_ids:
                save_run_journal(
                    cache_root,
                    RunJournalRecord(stage="embeddings_and_captions", cursor_image_id=batch_ids[-1], updated_at=time.time()),
                )

        # Captions
        caption_batch_size = max(1, caption_cfg.batch_size)
        total_captions = 0

        from PIL import Image as _ImageCaption

        for batch_start in range(0, len(caption_targets), caption_batch_size):
            batch_ids = caption_targets[batch_start : batch_start + caption_batch_size]
            images: List["_ImageCaption.Image"] = []
            valid_ids: List[str] = []

            for image_id in batch_ids:
                path_str = paths_by_id[image_id]
                try:
                    img = _ImageCaption.open(path_str).convert("RGB")
                except Exception as exc:
                    self._logger.error(
                        "caption_image_open_error",
                        extra={"image_id": image_id, "path": path_str, "error": str(exc)},
                    )
                    continue

                images.append(img)
                valid_ids.append(image_id)

            if not images:
                continue

            inputs = blip_processor(images=images, return_tensors="pt")
            inputs = inputs.to(blip_device)

            with torch.no_grad():
                generated_ids = blip_model.generate(**inputs, padding="max_length")

            for idx, image_id in enumerate(valid_ids):
                token_ids = generated_ids[idx]
                caption_text = blip_processor.decode(token_ids, skip_special_tokens=True)

                rel_path = f"{image_id}.json"
                caption_path = captions_dir / rel_path
                payload = {
                    "image_id": image_id,
                    "caption": caption_text,
                    "model_name": caption_model_name,
                    "model_backend": caption_cfg.backend,
                    "updated_at": now,
                }
                caption_path.write_text(json.dumps(payload), encoding="utf-8")

                primary_cursor.execute(
                    """
                    INSERT INTO image_caption (
                        image_id,
                        model_name,
                        caption,
                        model_backend,
                        updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(image_id, model_name) DO UPDATE SET
                        caption = excluded.caption,
                        model_backend = excluded.model_backend,
                        updated_at = excluded.updated_at
                    """,
                    (
                        image_id,
                        caption_model_name,
                        caption_text,
                        caption_cfg.backend,
                        now,
                    ),
                )
                total_captions += 1

        primary_conn.commit()
        self._logger.info(
            "embeddings_and_captions_complete",
            extra={"embeddings_created": total_embeddings, "captions_created": total_captions},
        )

    def _run_perceptual_hashing_and_duplicates(self, primary_conn, projection_conn) -> None:
        """Compute perceptual hashes and rebuild near-duplicate groups."""

        self._logger.info("phash_and_duplicates_start", extra={})

        cursor = primary_conn.cursor()
        now = time.time()

        cursor.execute(
            """
            SELECT image_id, primary_path
            FROM images
            WHERE status = 'active'
              AND (
                phash IS NULL
                OR phash_algo IS NULL
                OR phash_algo != ?
                OR width IS NULL
                OR height IS NULL
              )
            """,
            (PHASH_ALGO,),
        )
        rows = cursor.fetchall()

        phash_updated = 0
        for row in rows:
            image_id = row["image_id"]
            path_str = row["primary_path"]
            try:
                from PIL import Image

                image = Image.open(path_str)
                width, height = image.size
            except Exception as exc:
                self._logger.error("phash_image_open_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
                cursor.execute(
                    """
                    UPDATE images
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE image_id = ?
                    """,
                    ("error", f"phash_image_open_error: {exc}", now, image_id),
                )
                continue

            try:
                phash_hex = compute_perceptual_hash(image)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("phash_compute_error", extra={"image_id": image_id, "path": path_str, "error": str(exc)})
                cursor.execute(
                    """
                    UPDATE images
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE image_id = ?
                    """,
                    ("error", f"phash_compute_error: {exc}", now, image_id),
                )
                continue

            cursor.execute(
                """
                UPDATE images
                SET width = ?, height = ?, phash = ?, phash_algo = ?, phash_updated_at = ?, status = ?, error_message = NULL, updated_at = ?
                WHERE image_id = ?
                """,
                (int(width), int(height), phash_hex, PHASH_ALGO, now, "active", now, image_id),
            )
            phash_updated += 1

        primary_conn.commit()
        self._logger.info("phash_update_complete", extra={"updated_count": phash_updated})

        # Build near-duplicate groups using prefix buckets on the high-order bits.
        cursor.execute(
            """
            SELECT image_id, phash
            FROM images
            WHERE status = 'active' AND phash IS NOT NULL AND phash_algo = ?
            """,
            (PHASH_ALGO,),
        )
        rows = cursor.fetchall()

        BUCKET_PREFIX_HEX_LEN = 4  # 16 high bits
        HAMMING_THRESHOLD = 12

        buckets: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            phash_hex = row["phash"]
            if not phash_hex or len(phash_hex) < BUCKET_PREFIX_HEX_LEN:
                continue
            prefix = phash_hex[:BUCKET_PREFIX_HEX_LEN]
            buckets.setdefault(prefix, []).append({"image_id": row["image_id"], "phash": phash_hex})

        near_pairs: List[tuple[str, str, int]] = []
        for prefix, items in buckets.items():
            items_sorted = sorted(items, key=lambda item: item["image_id"])
            used: set[str] = set()
            for i, anchor in enumerate(items_sorted):
                anchor_id = anchor["image_id"]
                if anchor_id in used:
                    continue
                anchor_phash = anchor["phash"]
                for candidate in items_sorted[i + 1 :]:
                    candidate_id = candidate["image_id"]
                    if candidate_id in used:
                        continue
                    distance = hamming_distance_phash(anchor_phash, candidate["phash"])
                    if distance <= HAMMING_THRESHOLD:
                        near_pairs.append((anchor_id, candidate_id, distance))
                        used.add(candidate_id)
                used.add(anchor_id)

        proj_cursor = primary_conn.cursor()
        proj_cursor.execute("DELETE FROM image_near_duplicate")
        for anchor_id, duplicate_id, distance in near_pairs:
            proj_cursor.execute(
                """
                INSERT INTO image_near_duplicate (anchor_image_id, duplicate_image_id, phash_distance, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (anchor_id, duplicate_id, distance, now),
            )
        primary_conn.commit()

        self._logger.info(
            "phash_and_duplicates_complete",
            extra={"phash_updated": phash_updated, "near_duplicate_pairs": len(near_pairs)},
        )


__all__ = ["PreprocessingPipeline", "RunJournalRecord", "load_run_journal", "save_run_journal"]
