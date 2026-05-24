"""Benchmark OpenCV ECC vs ECC multiscale on correction-history images.

Run this with an OpenCV build that exposes ``cv2.findTransformECCMultiScale``.
For the local experiment that motivated this script:

    /tmp/cv-ecc-dev311/bin/python examples/bench_ecc_multiscale.py \
        --calibration /Users/khan/Downloads/calibration_200___.h5 \
        --history /Users/khan/Downloads/calibration_200____corrections.h5 \
        --output-dir /tmp/opencv-ecc-ms/bench_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


CAMERAS = ("cam0", "cam1")
PIXEL_AXES = ("du_px", "dv_px")


@dataclass(frozen=True)
class ImageRecord:
    run: str
    camera: str
    image: np.ndarray
    shift_px: np.ndarray


@dataclass(frozen=True)
class Result:
    family: str
    case_id: str
    camera: str
    motion_model: str
    algorithm: str
    seed: str
    expected_du_px: float
    expected_dv_px: float
    measured_du_px: float
    measured_dv_px: float
    error_px: float
    warp_error: float
    correlation: float
    elapsed_ms: float
    success: bool
    message: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-synthetic-bases", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration_reference, calibration_attrs = load_calibration_reference(
        args.calibration
    )
    reference_crops = {
        camera: crop_to_roi(
            calibration_reference[camera],
            calibration_attrs,
            camera,
        )
        for camera in CAMERAS
    }
    records = load_history_records(args.history)

    verify_cv2(args.output_dir)

    results: list[Result] = []
    for motion_model in ("affine", "homography"):
        results.extend(
            run_reference_history_cases(reference_crops, records, motion_model)
        )
        results.extend(run_adjacent_history_cases(records, motion_model))
        results.extend(
            run_synthetic_cases(
                reference_crops,
                records,
                motion_model,
                max_bases=args.max_synthetic_bases,
            )
        )

    write_results(results, args.output_dir / "ecc_multiscale_results.csv")
    write_summary(results, args.output_dir)
    make_plots(results, args.output_dir)
    make_visual_checks(reference_crops, records, results, args.output_dir)


def verify_cv2(output_dir: Path) -> None:
    info = {
        "cv2_version": cv2.__version__,
        "has_findTransformECC": hasattr(cv2, "findTransformECC"),
        "has_findTransformECCMultiScale": hasattr(
            cv2,
            "findTransformECCMultiScale",
        ),
        "has_ECCParameters": hasattr(cv2, "ECCParameters"),
    }
    if not info["has_findTransformECCMultiScale"] or not info["has_ECCParameters"]:
        raise RuntimeError(f"OpenCV build lacks multiscale ECC API: {info}")
    with (output_dir / "cv2_info.json").open("w") as file:
        json.dump(info, file, indent=2)


def load_calibration_reference(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with h5py.File(path, "r") as handle:
        attrs = {key: scalar_attr(value) for key, value in handle.attrs.items()}
        refs = {camera: handle[f"reference_{camera}"][()] for camera in CAMERAS}
    return refs, attrs


def load_history_records(path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with h5py.File(path, "r") as handle:
        for run in sorted(handle):
            group = handle[run]
            shifts = np.asarray(group["shift_px"], dtype=np.float64)
            for camera_index, camera in enumerate(CAMERAS):
                records.append(
                    ImageRecord(
                        run=run,
                        camera=camera,
                        image=group[f"current_{camera}"][()],
                        shift_px=shifts[camera_index].copy(),
                    )
                )
    return records


def scalar_attr(value: object) -> object:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def crop_to_roi(
    image: np.ndarray,
    attrs: dict[str, object],
    camera: str,
) -> np.ndarray:
    x = float(attrs[f"roi_{camera}_x"])
    y = float(attrs[f"roi_{camera}_y"])
    width = float(attrs[f"roi_{camera}_width"])
    height = float(attrs[f"roi_{camera}_height"])
    image_height, image_width = image.shape[:2]
    width = min(max(width, 1.0), float(image_width))
    height = min(max(height, 1.0), float(image_height))
    x = min(max(x, 0.0), float(image_width) - width)
    y = min(max(y, 0.0), float(image_height) - height)
    x0 = min(max(int(math.floor(x)), 0), image_width - 1)
    y0 = min(max(int(math.floor(y)), 0), image_height - 1)
    x1 = min(max(int(math.ceil(x + width)), x0 + 1), image_width)
    y1 = min(max(int(math.ceil(y + height)), y0 + 1), image_height)
    return image[y0:y1, x0:x1]


def run_reference_history_cases(
    reference_crops: dict[str, np.ndarray],
    records: list[ImageRecord],
    motion_model: str,
) -> list[Result]:
    results: list[Result] = []
    for record in records:
        reference = reference_crops[record.camera]
        case_id = f"{record.run}:{record.camera}:reference"
        expected = record.shift_px
        for algorithm in ("phase_only", "single_ecc", "multi_ecc"):
            results.append(
                evaluate_pair(
                    reference,
                    record.image,
                    expected,
                    family="reference_history",
                    case_id=case_id,
                    camera=record.camera,
                    motion_model=motion_model,
                    algorithm=algorithm,
                    seed="phase",
                )
            )
    return results


def run_adjacent_history_cases(
    records: list[ImageRecord],
    motion_model: str,
) -> list[Result]:
    results: list[Result] = []
    by_camera = {camera: [] for camera in CAMERAS}
    for record in records:
        by_camera[record.camera].append(record)
    for camera, camera_records in by_camera.items():
        camera_records.sort(key=lambda record: record.run)
        for left, right in zip(camera_records, camera_records[1:], strict=False):
            expected = right.shift_px - left.shift_px
            case_id = f"{left.run}_to_{right.run}:{camera}"
            for algorithm in ("phase_only", "single_ecc", "multi_ecc"):
                results.append(
                    evaluate_pair(
                        left.image,
                        right.image,
                        expected,
                        family="adjacent_history",
                        case_id=case_id,
                        camera=camera,
                        motion_model=motion_model,
                        algorithm=algorithm,
                        seed="phase",
                    )
                )
    return results


def run_synthetic_cases(
    reference_crops: dict[str, np.ndarray],
    records: list[ImageRecord],
    motion_model: str,
    *,
    max_bases: int,
) -> list[Result]:
    results: list[Result] = []
    bases = synthetic_bases(reference_crops, records, max_bases)
    for camera, label, image in bases:
        for transform_name, warp in synthetic_transforms(image.shape):
            sample = warp_image(image, warp)
            expected = point_shift(warp, image_center(image.shape))
            case_id = f"{label}:{transform_name}"
            for seed in ("phase", "identity"):
                for algorithm in ("single_ecc", "multi_ecc"):
                    results.append(
                        evaluate_pair(
                            image,
                            sample,
                            expected,
                            family="synthetic",
                            case_id=case_id,
                            camera=camera,
                            motion_model=motion_model,
                            algorithm=algorithm,
                            seed=seed,
                        )
                    )
    return results


def synthetic_bases(
    reference_crops: dict[str, np.ndarray],
    records: list[ImageRecord],
    max_bases: int,
) -> list[tuple[str, str, np.ndarray]]:
    bases: list[tuple[str, str, np.ndarray]] = [
        (camera, f"calibration_reference_{camera}", reference_crops[camera])
        for camera in CAMERAS
    ]
    by_camera = {camera: [] for camera in CAMERAS}
    for record in records:
        by_camera[record.camera].append(record)
    per_camera = max(1, (max_bases - len(bases)) // len(CAMERAS))
    for camera, camera_records in by_camera.items():
        camera_records.sort(key=lambda record: np.linalg.norm(record.shift_px))
        indexes = np.linspace(0, len(camera_records) - 1, per_camera, dtype=int)
        for index in sorted(set(indexes)):
            record = camera_records[index]
            bases.append((camera, record.run + f"_{camera}", record.image))
    return bases[:max_bases]


def synthetic_transforms(
    shape: tuple[int, int],
) -> list[tuple[str, np.ndarray]]:
    center = image_center(shape)
    transforms: list[tuple[str, np.ndarray]] = []
    for name, dx, dy in [
        ("tiny_translation", 0.12, -0.08),
        ("small_translation", -0.45, 0.35),
        ("medium_translation", 1.6, -1.1),
        ("large_translation", -4.5, 3.2),
    ]:
        transforms.append((name, translation_warp(dx, dy)))
    transforms.extend(
        [
            (
                "small_rotation",
                affine_about_center(center, angle_deg=0.25, scale=1.0, shift=(0.4, -0.2)),
            ),
            (
                "moderate_rotation",
                affine_about_center(center, angle_deg=1.25, scale=1.0, shift=(-1.2, 0.9)),
            ),
            (
                "small_scale",
                affine_about_center(center, angle_deg=0.0, scale=1.01, shift=(0.6, -0.3)),
            ),
            (
                "scale_rotation",
                affine_about_center(center, angle_deg=-1.0, scale=0.985, shift=(-1.0, 1.3)),
            ),
            (
                "mild_shear",
                np.asarray(
                    [
                        [1.0, 0.012, -0.7],
                        [-0.006, 1.0, 0.8],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                "mild_homography",
                np.asarray(
                    [
                        [1.0, 0.010, -1.2],
                        [-0.004, 0.995, 0.9],
                        [8e-5, -6e-5, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return transforms


def translation_warp(dx: float, dy: float) -> np.ndarray:
    return np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


def affine_about_center(
    center: np.ndarray,
    *,
    angle_deg: float,
    scale: float,
    shift: tuple[float, float],
) -> np.ndarray:
    angle = math.radians(angle_deg)
    cosine = scale * math.cos(angle)
    sine = scale * math.sin(angle)
    matrix = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
        dtype=np.float64,
    )
    matrix[:, 2] = center - matrix[:, :2] @ center + np.asarray(shift)
    return matrix.astype(np.float32)


def warp_image(image: np.ndarray, warp: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    work = image.astype(np.float32)
    if warp.shape == (2, 3):
        return cv2.warpAffine(
            work,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return cv2.warpPerspective(
        work,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def evaluate_pair(
    reference: np.ndarray,
    sample: np.ndarray,
    expected_shift: np.ndarray,
    *,
    family: str,
    case_id: str,
    camera: str,
    motion_model: str,
    algorithm: str,
    seed: str,
) -> Result:
    start = time.perf_counter()
    try:
        reference_norm = normalize_intensity(reference)
        sample_norm = normalize_intensity(sample)
        if seed == "identity":
            phase_shift = np.zeros(2, dtype=np.float64)
        else:
            phase_shift = phase_correlate(reference_norm, sample_norm)
        if algorithm == "phase_only":
            measured = phase_shift
            warp = initial_warp(motion_model, phase_shift)
            corr = normalized_correlation(reference_norm, sample_norm, warp)
        else:
            measured, warp, corr = ecc_refine(
                reference_norm,
                sample_norm,
                phase_shift,
                motion_model=motion_model,
                algorithm=algorithm,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        error_px = float(np.linalg.norm(measured - expected_shift))
        return Result(
            family=family,
            case_id=case_id,
            camera=camera,
            motion_model=motion_model,
            algorithm=algorithm,
            seed=seed,
            expected_du_px=float(expected_shift[0]),
            expected_dv_px=float(expected_shift[1]),
            measured_du_px=float(measured[0]),
            measured_dv_px=float(measured[1]),
            error_px=error_px,
            warp_error=warp_sanity_error(warp),
            correlation=float(corr),
            elapsed_ms=elapsed_ms,
            success=True,
            message="",
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Result(
            family=family,
            case_id=case_id,
            camera=camera,
            motion_model=motion_model,
            algorithm=algorithm,
            seed=seed,
            expected_du_px=float(expected_shift[0]),
            expected_dv_px=float(expected_shift[1]),
            measured_du_px=float("nan"),
            measured_dv_px=float("nan"),
            error_px=float("nan"),
            warp_error=float("nan"),
            correlation=float("nan"),
            elapsed_ms=elapsed_ms,
            success=False,
            message=str(exc).replace("\n", " ")[:400],
        )


def normalize_intensity(
    image: np.ndarray,
    *,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    work = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(work, clip_percentiles)
    if high > low:
        work = np.clip(work, low, high)
    work = work - np.mean(work)
    std = np.std(work)
    if std <= 1e-12:
        return np.zeros_like(work, dtype=np.float32)
    return np.ascontiguousarray(work / std, dtype=np.float32)


def phase_correlate(reference: np.ndarray, sample: np.ndarray) -> np.ndarray:
    shift = cv2.phaseCorrelateIterative(
        np.ascontiguousarray(reference, dtype=np.float32),
        np.ascontiguousarray(sample, dtype=np.float32),
        7,
        50,
    )
    return np.asarray(shift, dtype=np.float64)


def ecc_refine(
    reference: np.ndarray,
    sample: np.ndarray,
    initial_shift_px: np.ndarray,
    *,
    motion_model: str,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    motion_code = motion_code_for_model(motion_model)
    warp = initial_warp(motion_model, initial_shift_px)
    reference_work = np.ascontiguousarray(reference, dtype=np.float32)
    sample_work = np.ascontiguousarray(sample, dtype=np.float32)
    if algorithm == "single_ecc":
        corr, refined = cv2.findTransformECC(
            reference_work,
            sample_work,
            warp,
            motion_code,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
            None,
            5,
        )
    elif algorithm == "multi_ecc":
        params = cv2.ECCParameters()
        params.motionType = motion_code
        params.nlevels = 4
        params.gaussFiltSize = 5
        params.interpolation = cv2.INTER_LINEAR
        corr, refined = cv2.findTransformECCMultiScale(
            reference_work,
            sample_work,
            warp,
            params,
        )
    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    refined = np.asarray(refined, dtype=np.float64)
    shift = point_shift(refined, image_center(reference.shape))
    return shift, refined, float(corr)


def motion_code_for_model(motion_model: str) -> int:
    if motion_model == "affine":
        return cv2.MOTION_AFFINE
    if motion_model == "homography":
        return cv2.MOTION_HOMOGRAPHY
    raise ValueError(motion_model)


def initial_warp(motion_model: str, shift: np.ndarray) -> np.ndarray:
    if motion_model == "affine":
        return np.asarray(
            [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]],
            dtype=np.float32,
        )
    if motion_model == "homography":
        return np.asarray(
            [
                [1.0, 0.0, shift[0]],
                [0.0, 1.0, shift[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    raise ValueError(motion_model)


def image_center(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape[:2]
    return np.asarray([(width - 1.0) / 2.0, (height - 1.0) / 2.0])


def point_shift(warp: np.ndarray, point: np.ndarray) -> np.ndarray:
    warp64 = np.asarray(warp, dtype=np.float64)
    if warp64.shape == (2, 3):
        return warp64 @ np.r_[point, 1.0] - point
    mapped = warp64 @ np.r_[point, 1.0]
    return mapped[:2] / mapped[2] - point


def normalized_correlation(
    reference: np.ndarray,
    sample: np.ndarray,
    warp: np.ndarray,
) -> float:
    warped = warp_sample_to_reference(sample, warp)
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(warped, dtype=np.float64).ravel()
    a -= np.mean(a)
    b -= np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def warp_sample_to_reference(sample: np.ndarray, warp: np.ndarray) -> np.ndarray:
    height, width = sample.shape[:2]
    if warp.shape == (2, 3):
        return cv2.warpAffine(
            sample.astype(np.float32),
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return cv2.warpPerspective(
        sample.astype(np.float32),
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def warp_sanity_error(warp: np.ndarray) -> float:
    if not np.isfinite(warp).all():
        return float("inf")
    if warp.shape == (2, 3):
        linear = warp[:, :2]
    else:
        linear = warp[:2, :2] / warp[2, 2]
    return float(np.linalg.norm(linear - np.eye(2)))


def write_results(results: list[Result], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_summary(results: list[Result], output_dir: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[Result]] = {}
    for result in results:
        key = (
            result.family,
            result.camera,
            result.motion_model,
            result.algorithm,
            result.seed,
        )
        groups.setdefault(key, []).append(result)

    rows = []
    for key, group in sorted(groups.items()):
        successes = [result for result in group if result.success]
        errors = [result.error_px for result in successes if np.isfinite(result.error_px)]
        elapsed = [result.elapsed_ms for result in successes]
        correlations = [
            result.correlation
            for result in successes
            if np.isfinite(result.correlation)
        ]
        rows.append(
            {
                "family": key[0],
                "camera": key[1],
                "motion_model": key[2],
                "algorithm": key[3],
                "seed": key[4],
                "n": len(group),
                "success_rate": len(successes) / len(group) if group else float("nan"),
                "median_error_px": median(errors),
                "p95_error_px": percentile(errors, 95),
                "median_elapsed_ms": median(elapsed),
                "median_correlation": median(correlations),
            }
        )

    with (output_dir / "summary_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cv2_version": cv2.__version__,
        "total_results": len(results),
        "groups": rows,
        "worst_successes": [
            asdict(result)
            for result in sorted(
                (r for r in results if r.success and np.isfinite(r.error_px)),
                key=lambda item: item.error_px,
                reverse=True,
            )[:20]
        ],
        "failures": [asdict(result) for result in results if not result.success][:50],
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "summary.md").open("w") as file:
        file.write(f"# ECC multiscale benchmark\n\nOpenCV: `{cv2.__version__}`\n\n")
        file.write("| family | camera | model | algorithm | seed | n | success | median error px | p95 error px | median ms | median corr |\n")
        file.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                "| {family} | {camera} | {motion_model} | {algorithm} | {seed} | "
                "{n} | {success_rate:.3f} | {median_error_px:.4g} | "
                "{p95_error_px:.4g} | {median_elapsed_ms:.4g} | "
                "{median_correlation:.4g} |\n".format(**row)
            )


def make_plots(results: list[Result], output_dir: Path) -> None:
    for family in sorted({result.family for result in results}):
        family_results = [r for r in results if r.family == family and r.success]
        labels = []
        values = []
        for algorithm in ("phase_only", "single_ecc", "multi_ecc"):
            for model in ("affine", "homography"):
                subset = [
                    r.error_px
                    for r in family_results
                    if r.algorithm == algorithm
                    and r.motion_model == model
                    and r.seed == "phase"
                    and np.isfinite(r.error_px)
                ]
                if subset:
                    labels.append(f"{algorithm}\n{model}")
                    values.append(subset)
        if not values:
            continue
        plt.figure(figsize=(max(8, len(values) * 1.2), 4.5))
        plt.boxplot(values, tick_labels=labels, showfliers=False)
        plt.ylabel("error at reference point (px)")
        plt.title(f"{family}: error distribution")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{family}_error_boxplot.png", dpi=180)
        plt.close()


def make_visual_checks(
    reference_crops: dict[str, np.ndarray],
    records: list[ImageRecord],
    results: list[Result],
    output_dir: Path,
) -> None:
    image_dir = output_dir / "visual_checks"
    image_dir.mkdir(exist_ok=True)
    records_by_key = {(record.run, record.camera): record for record in records}
    selected = [
        result
        for result in sorted(
            (
                r
                for r in results
                if r.family == "reference_history"
                and r.algorithm in {"single_ecc", "multi_ecc"}
                and r.seed == "phase"
                and r.success
                and np.isfinite(r.error_px)
            ),
            key=lambda item: item.error_px,
            reverse=True,
        )[:8]
    ]
    for result in selected:
        run = result.case_id.split(":", 1)[0]
        record = records_by_key[(run, result.camera)]
        reference = normalize_intensity(reference_crops[result.camera])
        sample = normalize_intensity(record.image)
        initial = np.asarray([result.measured_du_px, result.measured_dv_px])
        try:
            _, warp, _ = ecc_refine(
                reference,
                sample,
                initial,
                motion_model=result.motion_model,
                algorithm=result.algorithm,
            )
        except Exception:
            continue
        warped = warp_sample_to_reference(sample, warp)
        diff = np.abs(reference - warped)
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        panels = [
            ("reference", reference),
            ("sample", sample),
            ("warped sample", warped),
            ("abs diff", diff),
        ]
        for axis, (title, image) in zip(axes, panels, strict=True):
            axis.imshow(image, cmap="gray")
            axis.set_title(title)
            axis.set_axis_off()
        fig.suptitle(
            f"{result.case_id} {result.motion_model}/{result.algorithm} "
            f"err={result.error_px:.3g}px corr={result.correlation:.3g}",
            fontsize=10,
        )
        fig.tight_layout()
        safe = result.case_id.replace(":", "_").replace("/", "_")
        fig.savefig(
            image_dir / f"{safe}_{result.motion_model}_{result.algorithm}.png",
            dpi=180,
        )
        plt.close(fig)


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, pct))


if __name__ == "__main__":
    main()
