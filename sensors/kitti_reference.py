"""Evaluation-only KITTI odometry reference loading and UTM alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KittiReferenceTrajectory:
    positions_local_xz_m: np.ndarray
    poses_world_camera: np.ndarray
    source_file: Path


def load_kitti_odometry_poses(path: Path) -> KittiReferenceTrajectory:
    """Read KITTI's 3x4-per-line odometry ground-truth pose format."""
    source = Path(path).expanduser().resolve()
    values = np.loadtxt(source, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 12:
        raise ValueError(
            f"Expected 12 values per KITTI pose row, got {values.shape[1]} in {source}"
        )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(values), axis=0)
    poses[:, :3, :] = values.reshape(-1, 3, 4)
    # KITTI odometry poses use the camera coordinate convention. Horizontal
    # motion is represented by camera x (right) and z (forward).
    positions = poses[:, (0, 2), 3]
    return KittiReferenceTrajectory(positions, poses, source)


def rigid_align_reference_to_utm(
    reference_local_xy: np.ndarray,
    initial_utm_xy: np.ndarray,
    *,
    alignment_frames: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rigidly align a metric local reference to UTM using only its beginning.

    No scale is estimated because both KITTI odometry poses and UTM are metric.
    The returned RMS describes only the short alignment segment and is not an
    estimator-quality metric.
    """
    source = np.asarray(reference_local_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(initial_utm_xy, dtype=np.float64).reshape(-1, 2)
    count = min(int(alignment_frames), len(source), len(target))
    if count < 3:
        raise ValueError("At least three frames are required for UTM alignment")
    source_fit = source[:count]
    target_fit = target[:count]
    source_center = np.mean(source_fit, axis=0)
    target_center = np.mean(target_fit, axis=0)
    source_centered = source_fit - source_center
    target_centered = target_fit - target_center
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cross_covariance = source_centered.T @ target_centered
    if not np.all(np.isfinite(cross_covariance)):
        raise FloatingPointError("Non-finite KITTI-to-UTM alignment covariance")
    u, _singular_values, vt = np.linalg.svd(cross_covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        aligned = (rotation @ source.T).T + translation
    if not np.all(np.isfinite(aligned)):
        raise FloatingPointError("Non-finite aligned KITTI reference trajectory")
    residual = aligned[:count] - target_fit
    alignment_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return aligned, rotation, translation, alignment_rmse


def reference_for_selected_frames(
    reference: KittiReferenceTrajectory,
    frame_indices: np.ndarray,
    selected_utm_xy: np.ndarray,
    *,
    alignment_frames: int = 20,
) -> tuple[np.ndarray, float]:
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= len(reference.positions_local_xz_m)):
        raise IndexError(
            f"Selected KITTI frames exceed reference range 0.."
            f"{len(reference.positions_local_xz_m) - 1}"
        )
    selected_reference = reference.positions_local_xz_m[indices]
    aligned, _rotation, _translation, alignment_rmse = rigid_align_reference_to_utm(
        selected_reference,
        selected_utm_xy,
        alignment_frames=alignment_frames,
    )
    return aligned, alignment_rmse
