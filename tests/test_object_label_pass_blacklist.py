from __future__ import annotations

from pathlib import Path

import numpy as np
from sqlalchemy import select

from vibe_photos.config import Settings
from vibe_photos.db import (
    Label,
    LabelAssignment,
    Region,
    RegionEmbedding,
    open_primary_session,
    open_projection_session,
)
from vibe_photos.labels.object_label_pass import run_object_label_pass
from vibe_photos.labels.repository import LabelRepository
from vibe_photos.labels.seed_labels import seed_labels


def _write_proto(cache_root: Path, label_ids: list[int]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    proto_dir = cache_root / "label_text_prototypes"
    proto_dir.mkdir(parents=True, exist_ok=True)

    vecs = []
    for i in range(len(label_ids)):
        v = np.zeros(4, dtype=np.float32)
        v[i % 4] = 1.0
        vecs.append(v)

    np.savez(
        proto_dir / "object_v1.npz",
        label_ids=np.asarray(label_ids),
        label_keys=np.asarray(label_ids),
        prototypes=np.stack(vecs, axis=0),
    )


def test_object_pass_applies_blacklist_and_remap(tmp_path: Path) -> None:
    data_db = tmp_path / "data.db"
    cache_db = tmp_path / "cache.db"
    cache_root = tmp_path / "cache"

    settings = Settings()
    settings.label_spaces.object_current = "object_v1"
    settings.label_spaces.scene_current = "scene_v1"
    settings.object.zero_shot.scene_whitelist = ["scene.product"]

    with open_primary_session(data_db) as primary, open_projection_session(cache_db) as projection:
        seed_labels(primary)
        primary.commit()

        repo = LabelRepository(primary)
        label_old = repo.get_or_create_label(key="object.test.old", level="object", display_name="Old")
        label_new = repo.get_or_create_label(key="object.test.new", level="object", display_name="New")
        primary.commit()

        # Scene gate to allow labeling
        scene_label = primary.execute(select(Label).where(Label.key == "scene.product")).scalar_one()
        primary.add(
            LabelAssignment(
                target_type="image",
                target_id="img1",
                label_id=scene_label.id,
                source="classifier",
                label_space_ver="scene_v1",
                score=0.9,
                extra_json=None,
                created_at=1.0,
                updated_at=1.0,
            )
        )
        primary.commit()

        # Region + embedding aligned to old label
        projection.add(
            Region(
                id="img1#0",
                image_id="img1",
                x_min=0.0,
                y_min=0.0,
                x_max=1.0,
                y_max=1.0,
                detector="owlvit",
                raw_label="thing",
                raw_score=0.9,
                created_at=1.0,
            )
        )
        emb_rel = f"regions/{settings.models.embedding.resolved_model_name()}/img1#0.npy"
        emb_path = cache_root / "embeddings" / emb_rel
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        vec = np.zeros(4, dtype=np.float32)
        vec[0] = 1.0
        np.save(emb_path, vec)
        projection.add(
            RegionEmbedding(
                region_id="img1#0",
                model_name=settings.models.embedding.resolved_model_name(),
                embedding_path=emb_rel,
                embedding_dim=4,
                backend=settings.models.embedding.backend,
                updated_at=1.0,
            )
        )
        projection.commit()

        _write_proto(cache_root, [label_old.id, label_new.id])

        settings.object.blacklist = ["object.test.new"]
        settings.object.remap = {"object.test.old": "object.test.new"}

        run_object_label_pass(
            primary_session=primary,
            projection_session=projection,
            settings=settings,
            cache_root=cache_root,
            label_space_ver="object_v1",
            prototype_name="object_v1",
        )

        rows = primary.execute(
            select(Label.display_name, Label.key)
            .join(LabelAssignment, Label.id == LabelAssignment.label_id)
            .where(
                LabelAssignment.target_type == "image",
                LabelAssignment.target_id == "img1",
                LabelAssignment.label_space_ver == "object_v1",
            )
        ).all()

        assigned_keys = {row.key for row in rows}

        assert "object.test.old" not in assigned_keys
        assert "object.test.new" in assigned_keys
