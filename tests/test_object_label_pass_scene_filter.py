from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sqlalchemy import select

from tests.utils.postgres import temporary_postgres
from vibe_photos.config import Settings
from vibe_photos.db import (
    Label,
    LabelAssignment,
    Region,
    RegionEmbedding,
    open_primary_session,
)
from vibe_photos.labels.object_label_pass import run_object_label_pass
from vibe_photos.labels.seed_labels import seed_labels


def _make_proto(cache_root: Path, label_ids: list[int], label_keys: list[str]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    proto_dir = cache_root / "label_text_prototypes"
    proto_dir.mkdir(parents=True, exist_ok=True)
    vecs = []
    for i in range(len(label_ids)):
        v = np.zeros(4, dtype=np.float32)
        v[i % 4] = 1.0
        vecs.append(v)
    np.savez(proto_dir / "object_v1.npz", label_ids=np.asarray(label_ids), label_keys=np.asarray(label_keys), prototypes=np.stack(vecs, axis=0))


def test_object_pass_respects_scene_filters(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    settings.label_spaces.object_current = "object_v1"
    settings.label_spaces.scene_current = "scene_v1"
    settings.object.zero_shot.scene_whitelist = ["scene.food"]
    settings.object.zero_shot.scene_fallback_labels = {"scene.screenshot": ["object.electronics.laptop"]}

    with temporary_postgres(tmp_path) as db_url, open_primary_session(db_url) as primary:
        cache_session = primary
        seed_labels(primary)
        primary.commit()

        laptop = primary.execute(select(Label).where(Label.key == "object.electronics.laptop")).scalar_one()
        phone = primary.execute(select(Label).where(Label.key == "object.electronics.phone")).scalar_one()

        # scene assignments
        now = 1.0
        primary.add(
            LabelAssignment(
                target_type="image",
                target_id="img_food",
                label_id=primary.execute(select(Label).where(Label.key == "scene.food")).scalar_one().id,
                source="classifier",
                label_space_ver="scene_v1",
                score=0.9,
                extra_json=json.dumps({"scene_margin": 1.0}),
                created_at=now,
                updated_at=now,
            )
        )
        primary.add(
            LabelAssignment(
                target_type="image",
                target_id="img_ss",
                label_id=primary.execute(select(Label).where(Label.key == "scene.screenshot")).scalar_one().id,
                source="classifier",
                label_space_ver="scene_v1",
                score=0.9,
                extra_json=json.dumps({"scene_margin": 1.0}),
                created_at=now,
                updated_at=now,
            )
        )
        primary.commit()

        # regions + embeddings
        for img_id in ("img_food", "img_ss"):
            region_id = f"{img_id}#0"
            cache_session.add(
                Region(
                    id=region_id,
                    image_id=img_id,
                    x_min=0.0,
                    y_min=0.0,
                    x_max=1.0,
                    y_max=1.0,
                    detector="owlvit",
                    raw_label="laptop",
                    raw_score=0.9,
                    created_at=now,
                )
            )
            emb_rel = f"regions/{settings.models.embedding.resolved_model_name()}/{region_id}.npy"
            emb_path = cache_root / "embeddings" / emb_rel
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            vec = np.zeros(4, dtype=np.float32)
            vec[0] = 1.0  # align with laptop prototype
            np.save(emb_path, vec)
            cache_session.add(
                RegionEmbedding(
                    region_id=region_id,
                    model_name=settings.models.embedding.resolved_model_name(),
                    embedding_path=emb_rel,
                    embedding_dim=4,
                    backend=settings.models.embedding.backend,
                    updated_at=now,
                )
            )
        cache_session.commit()

        # dummy stale assignment to ensure cleanup
        primary.add(
            LabelAssignment(
                target_type="image",
                target_id="img_food",
                label_id=phone.id,
                source="zero_shot",
                label_space_ver="object_v1",
                score=0.1,
                extra_json=None,
                created_at=0.0,
                updated_at=0.0,
            )
        )
        primary.commit()

        _make_proto(cache_root, [laptop.id, phone.id], [laptop.key, phone.key])

        run_object_label_pass(
            primary_session=primary,
            cache_session=cache_session,
            settings=settings,
            cache_root=cache_root,
            label_space_ver="object_v1",
            prototype_name="object_v1",
        )

        assignments = primary.execute(
            select(LabelAssignment).where(
                LabelAssignment.label_space_ver == "object_v1", LabelAssignment.target_type == "image"
            )
        ).scalars()
        by_image = {}
        for a in assignments:
            by_image.setdefault(a.target_id, []).append(a.label_id)

        # 'img_food' gets labeled (whitelist)
        assert laptop.id in by_image.get("img_food", [])
        # stale phone assignment removed
        assert phone.id not in by_image.get("img_food", [])
        # screenshot uses fallback set, so laptop still allowed
        assert laptop.id in by_image.get("img_ss", [])

        # no labels for non-whitelisted/non-fallback scenes
        assert "img_other" not in by_image
