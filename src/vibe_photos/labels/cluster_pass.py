"""Image and region clustering based on SigLIP embeddings."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.config import Settings
from vibe_photos.db import (
    ClusterMembership,
    ImageEmbedding,
    ImageNearDuplicateMembership,
    ImageSimilarityCluster,
    Label,
    LabelAssignment,
    Region,
    RegionEmbedding,
)
from vibe_photos.labels.repository import LabelRepository

LOGGER = get_logger(__name__, extra={"component": "cluster_pass"})


def run_image_cluster_pass(
    *,
    primary_session: Session,
    cache_session: Session,
    settings: Settings,
    cache_root: Path,
) -> tuple[int, int]:
    """Cluster canonical product/food images by SigLIP embeddings."""

    method = "siglip_image_knn_v1"
    params = settings.cluster.image
    scene_keys = ["scene.product", "scene.food"]
    candidate_ids = _load_scene_filtered_images(primary_session, settings, scene_keys)

    non_canonical_ids = {
        row.image_id
        for row in primary_session.execute(
            select(ImageNearDuplicateMembership.image_id).where(ImageNearDuplicateMembership.is_canonical.is_(False))
        )
    }
    target_ids = [image_id for image_id in candidate_ids if image_id not in non_canonical_ids]
    if len(target_ids) < int(params.min_size):
        LOGGER.info("image_cluster_skip", extra={"reason": "insufficient_candidates"})
        _reset_cluster_method(primary_session, method)
        return 0, 0

    embedding_model = settings.models.embedding.resolved_model_name()
    embedding_map = _load_image_embedding_paths(cache_session, embedding_model, target_ids)
    vectors, valid_ids = _load_vectors(cache_root, embedding_map)
    if len(valid_ids) < int(params.min_size):
        LOGGER.info("image_cluster_skip", extra={"reason": "insufficient_embeddings"})
        _reset_cluster_method(primary_session, method)
        return 0, 0

    graph = _build_knn_graph(valid_ids, vectors, int(params.k), float(params.sim_threshold))
    components = _extract_components(valid_ids, graph, int(params.min_size))
    if not components:
        LOGGER.info("image_cluster_no_components", extra={})
        _reset_cluster_method(primary_session, method)
        return 0, 0

    repo = LabelRepository(primary_session)
    _reset_cluster_method(primary_session, method)

    cluster_count, member_count = _write_clusters(
        primary_session=primary_session,
        repo=repo,
        method=method,
        target_type="image",
        label_space=settings.label_spaces.cluster_current,
        components=components,
        vector_by_id={image_id: vector for image_id, vector in zip(valid_ids, vectors, strict=True)},
        parent_lookup=None,
        params={
            "k": int(params.k),
            "sim_threshold": float(params.sim_threshold),
            "min_size": int(params.min_size),
        },
    )

    LOGGER.info(
        "image_cluster_complete",
        extra={"clusters": cluster_count, "members": member_count},
    )
    return cluster_count, member_count


def run_region_cluster_pass(
    *,
    primary_session: Session,
    cache_session: Session,
    settings: Settings,
    cache_root: Path,
) -> tuple[int, int]:
    """Cluster regions based on SigLIP embeddings."""

    method = "siglip_region_knn_v1"
    params = settings.cluster.region

    region_rows = cache_session.execute(
        select(
            RegionEmbedding.region_id,
            RegionEmbedding.embedding_path,
            Region.image_id,
        ).join(Region, Region.id == RegionEmbedding.region_id)
    )

    path_by_id: dict[str, tuple[str, str]] = {}
    for row in region_rows:
        path_by_id[row.region_id] = (row.embedding_path, row.image_id)

    vectors, valid_ids = _load_vectors(cache_root, {region_id: path for region_id, (path, _) in path_by_id.items()})
    if len(valid_ids) < int(params.min_size):
        LOGGER.info("region_cluster_skip", extra={"reason": "insufficient_embeddings"})
        _reset_cluster_method(primary_session, method)
        return 0, 0

    graph = _build_knn_graph(valid_ids, vectors, int(params.k), float(params.sim_threshold))
    components = _extract_components(valid_ids, graph, int(params.min_size))
    if not components:
        LOGGER.info("region_cluster_no_components", extra={})
        _reset_cluster_method(primary_session, method)
        return 0, 0

    parent_lookup = {region_id: path_by_id[region_id][1] for region_id in valid_ids if region_id in path_by_id}
    repo = LabelRepository(primary_session)
    _reset_cluster_method(primary_session, method)

    cluster_count, member_count = _write_clusters(
        primary_session=primary_session,
        repo=repo,
        method=method,
        target_type="region",
        label_space=settings.label_spaces.cluster_current,
        components=components,
        vector_by_id={region_id: vector for region_id, vector in zip(valid_ids, vectors, strict=True)},
        parent_lookup=parent_lookup,
        params={
            "k": int(params.k),
            "sim_threshold": float(params.sim_threshold),
            "min_size": int(params.min_size),
        },
    )

    LOGGER.info(
        "region_cluster_complete",
        extra={"clusters": cluster_count, "members": member_count},
    )
    return cluster_count, member_count


def _load_scene_filtered_images(primary_session: Session, settings: Settings, scene_keys: Sequence[str]) -> list[str]:
    scene_labels = {
        row.key: row.id
        for row in primary_session.execute(select(Label.id, Label.key).where(Label.key.in_(tuple(scene_keys))))
    }
    if not scene_labels:
        return []
    rows = primary_session.execute(
        select(LabelAssignment.target_id)
        .where(
            LabelAssignment.target_type == "image",
            LabelAssignment.label_space_ver == settings.label_spaces.scene_current,
            LabelAssignment.label_id.in_(scene_labels.values()),
        )
        .distinct()
    )
    return [row.target_id for row in rows]


def _load_image_embedding_paths(
    cache_session: Session, model_name: str, target_ids: Sequence[str]
) -> dict[str, str]:
    rows = cache_session.execute(
        select(ImageEmbedding.image_id, ImageEmbedding.embedding_path).where(ImageEmbedding.model_name == model_name)
    )
    mapping: dict[str, str] = {}
    for row in rows:
        if row.image_id in target_ids:
            mapping[row.image_id] = row.embedding_path
    return mapping


def _resolve_image_embedding_path(cache_root: Path, rel_path: str) -> Path:
    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return path_obj
    return cache_root / path_obj


def _load_vectors(cache_root: Path, path_map: dict[str, str]) -> tuple[list[np.ndarray], list[str]]:
    vectors: list[np.ndarray] = []
    valid_ids: list[str] = []
    for key, rel_path in path_map.items():
        emb_path = _resolve_image_embedding_path(cache_root, rel_path)
        try:
            vec = np.load(emb_path).astype(np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("cluster_embedding_load_error", extra={"target": key, "path": str(emb_path), "error": str(exc)})
            continue
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            continue
        vectors.append(vec / norm)
        valid_ids.append(key)
    return vectors, valid_ids


def _build_knn_graph(
    node_ids: Sequence[str],
    vectors: Sequence[np.ndarray],
    k: int,
    sim_threshold: float,
) -> dict[str, set[str]]:
    if not vectors:
        return {}

    node_ids = list(node_ids)
    matrix = np.stack(vectors)
    sim_matrix = matrix @ matrix.T
    np.fill_diagonal(sim_matrix, 0.0)

    graph: dict[str, set[str]] = defaultdict(set)
    for idx, sims in enumerate(sim_matrix):
        neighbors = np.argsort(sims)[::-1]
        added = 0
        for neighbor_idx in neighbors:
            if neighbor_idx == idx:
                continue
            if sims[neighbor_idx] < sim_threshold:
                continue
            graph[node_ids[idx]].add(node_ids[neighbor_idx])
            graph[node_ids[neighbor_idx]].add(node_ids[idx])
            added += 1
            if k > 0 and added >= k:
                break
    return graph


def _extract_components(
    node_ids: Sequence[str],
    graph: dict[str, set[str]],
    min_size: int,
) -> list[list[str]]:
    node_ids = list(node_ids)
    visited: set[str] = set()
    components: list[list[str]] = []

    for node in node_ids:
        if node in visited:
            continue
        stack = [node]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(graph.get(current, []))
        if len(component) >= min_size:
            components.append(component)
    return components


def _reset_cluster_method(primary_session: Session, method: str) -> None:
    cluster_rows = primary_session.execute(
        select(ImageSimilarityCluster.id, ImageSimilarityCluster.key).where(ImageSimilarityCluster.method == method)
    ).all()
    if not cluster_rows:
        return

    cluster_ids = [row.id for row in cluster_rows]
    cluster_keys = [row.key for row in cluster_rows if row.key]

    if cluster_keys:
        label_ids = set(
            primary_session.execute(select(Label.id).where(Label.key.in_(tuple(cluster_keys)))).scalars()
        )
        if label_ids:
            primary_session.execute(delete(LabelAssignment).where(LabelAssignment.label_id.in_(label_ids)))
        primary_session.execute(delete(Label).where(Label.key.in_(tuple(cluster_keys))))

    primary_session.execute(delete(ClusterMembership).where(ClusterMembership.cluster_id.in_(cluster_ids)))
    primary_session.execute(delete(ImageSimilarityCluster).where(ImageSimilarityCluster.id.in_(cluster_ids)))
    primary_session.commit()


def _write_clusters(
    *,
    primary_session: Session,
    repo: LabelRepository,
    method: str,
    target_type: str,
    label_space: str,
    components: Sequence[Sequence[str]],
    vector_by_id: dict[str, np.ndarray],
    parent_lookup: dict[str, str] | None,
    params: dict[str, object],
) -> tuple[int, int]:
    cluster_count = 0
    member_count = 0
    timestamp = time.time()

    for component in components:
        cluster_row = ImageSimilarityCluster(
            key="",
            method=method,
            params_json=json.dumps(params),
            created_at=timestamp,
        )
        primary_session.add(cluster_row)
        primary_session.flush()
        cluster_row.key = f"cluster.{method}.{cluster_row.id}"
        primary_session.add(cluster_row)
        primary_session.flush()

        label = repo.get_or_create_label(
            key=cluster_row.key,
            level="cluster",
            display_name=f"{method} #{cluster_row.id}",
        )

        center_id = _select_cluster_center(component, vector_by_id)
        for member_id in component:
            vector = vector_by_id.get(member_id)
            center_vec = vector_by_id.get(center_id) if center_id else None
            if vector is None or center_vec is None:
                continue
            distance = float(max(0.0, 1.0 - float(np.dot(center_vec, vector))))
            primary_session.add(
                ClusterMembership(
                    cluster_id=cluster_row.id,
                    target_type=target_type,
                    target_id=member_id,
                    distance=distance,
                    is_center=member_id == center_id,
                )
            )
            repo.upsert_label_assignment(
                target_type=target_type,
                target_id=member_id,
                label=label,
                source="cluster",
                label_space_ver=label_space,
                score=1.0,
                extra_json=json.dumps({"cluster_method": method}),
            )
            if target_type == "region" and parent_lookup is not None:
                image_id = parent_lookup.get(member_id)
                if image_id:
                    repo.upsert_label_assignment(
                        target_type="image",
                        target_id=image_id,
                        label=label,
                        source="cluster",
                        label_space_ver=label_space,
                        score=1.0,
                        extra_json=json.dumps({"cluster_method": method, "from_region": member_id}),
                    )
            member_count += 1

        cluster_count += 1

    primary_session.commit()
    return cluster_count, member_count


def _select_cluster_center(component: Sequence[str], vector_by_id: dict[str, np.ndarray]) -> str | None:
    best_id: str | None = None
    best_score: float = -1.0
    for member_id in component:
        vec = vector_by_id.get(member_id)
        if vec is None:
            continue
        score = 0.0
        for other_id in component:
            other_vec = vector_by_id.get(other_id)
            if other_vec is None:
                continue
            score += float(np.dot(vec, other_vec))
        if score > best_score:
            best_score = score
            best_id = member_id
    return best_id


__all__ = ["run_image_cluster_pass", "run_region_cluster_pass"]
