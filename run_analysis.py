#!/usr/bin/env python3
"""
Reproducible mosquito wing-beat-frequency analysis.

Analyses:
1. Free-flying female and male recordings:
   - candidate-WBF detection availability
   - recording-level median candidate WBF
   - recording-level median local peak prominence
   - accepted candidate-WBF distributions

2. Tethered female recordings:
   - candidate-WBF detection availability
   - recording-level median candidate WBF
   - recording-level median local peak prominence

3. Matched mosquito-absent ACL-2 ambient recordings:
   - power spectral density outside and inside the isolation enclosure
   - integrated ambient-power reduction over predefined frequency bands
   - one-third-octave ambient-power reduction

Important:
- Values are relative digital measurements, not calibrated dB SPL.
- Audio is converted to mono and DC offset is removed.
- Audio is not normalized, amplified, filtered, or denoised.
- Overlapping windows are summarized at the recording level.
- Candidate detections require peak prominence and harmonic support.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import find_peaks, welch


EPS = 1e-20


# =============================================================================
# CONFIGURATION AND FILE UTILITIES
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze mosquito wing-beat-frequency WAV recordings."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root input-data directory. Default: data",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory. Default: results",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="JSON configuration file. Default: config.json",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_wav_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    files = sorted(
        [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        ],
        key=natural_sort_key,
    )

    if not files:
        raise FileNotFoundError(f"No WAV files found under: {folder}")

    return files


def validate_single_file(path: Path, description: str):
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def make_output_directories(root: Path) -> dict[str, Path]:
    directories = {
        "root": root,
        "free_flight": root / "free_flight",
        "tethered": root / "tethered",
        "ambient": root / "ambient",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


# =============================================================================
# GENERAL AUDIO FUNCTIONS
# =============================================================================

def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a WAV file, average channels to mono, and remove DC offset.

    No normalization, filtering, amplification, or denoising is applied.
    """
    audio, sample_rate = sf.read(path, always_2d=True)

    audio = audio.astype(np.float64).mean(axis=1)
    audio -= np.mean(audio)

    return audio, int(sample_rate)


def power_to_db(power):
    return 10.0 * np.log10(np.maximum(power, EPS))


def calculate_psd(
    audio: np.ndarray,
    sample_rate: int,
    nperseg_requested: int,
) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg_requested, len(audio))

    if nperseg < 1024:
        raise ValueError(
            "Audio segment is too short for the requested spectral analysis."
        )

    frequencies, psd = welch(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    return frequencies, psd


def integrate_psd_band(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)

    if np.count_nonzero(mask) < 2:
        return np.nan

    return float(np.trapezoid(psd[mask], frequencies[mask]))


def relative_band_level_db(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    power = integrate_psd_band(frequencies, psd, low_hz, high_hz)

    if not np.isfinite(power) or power <= 0:
        return np.nan

    return float(power_to_db(power))


# =============================================================================
# CANDIDATE-WBF DETECTION
# =============================================================================

def local_peak_prominence_db(
    frequencies: np.ndarray,
    psd_db: np.ndarray,
    peak_index: int,
    local_background_half_width_hz: float,
    peak_exclusion_half_width_hz: float,
) -> float:
    f0 = frequencies[peak_index]

    local_mask = (
        (frequencies >= f0 - local_background_half_width_hz)
        & (frequencies <= f0 + local_background_half_width_hz)
    )

    exclusion_mask = (
        (frequencies >= f0 - peak_exclusion_half_width_hz)
        & (frequencies <= f0 + peak_exclusion_half_width_hz)
    )

    background = psd_db[local_mask & ~exclusion_mask]

    if background.size == 0:
        return np.nan

    return float(psd_db[peak_index] - np.median(background))


def harmonic_prominence_near_target(
    frequencies: np.ndarray,
    psd_db: np.ndarray,
    target_hz: float,
    tolerance_hz: float,
    local_background_half_width_hz: float,
    peak_exclusion_half_width_hz: float,
) -> tuple[float, float]:
    mask = (
        (frequencies >= target_hz - tolerance_hz)
        & (frequencies <= target_hz + tolerance_hz)
    )

    if not np.any(mask):
        return np.nan, np.nan

    candidate_indices = np.where(mask)[0]
    best_index = candidate_indices[np.argmax(psd_db[mask])]

    prominence = local_peak_prominence_db(
        frequencies=frequencies,
        psd_db=psd_db,
        peak_index=best_index,
        local_background_half_width_hz=local_background_half_width_hz,
        peak_exclusion_half_width_hz=peak_exclusion_half_width_hz,
    )

    return float(frequencies[best_index]), prominence


def empty_detection_result() -> dict:
    return {
        "detected": False,
        "f0_hz": np.nan,
        "primary_prominence_db": np.nan,
        "harmonic_count": 0,
        "harmonic_2_hz": np.nan,
        "harmonic_3_hz": np.nan,
        "harmonic_2_prominence_db": np.nan,
        "harmonic_3_prominence_db": np.nan,
    }


def estimate_candidate_wbf(
    segment: np.ndarray,
    sample_rate: int,
    search_low_hz: float,
    search_high_hz: float,
    config: dict,
) -> dict:
    detection = config["detection"]

    try:
        frequencies, psd = calculate_psd(
            segment,
            sample_rate,
            config["spectral"]["nperseg"],
        )
    except ValueError:
        return empty_detection_result()

    psd_db = power_to_db(psd)

    search_mask = (
        (frequencies >= search_low_hz)
        & (frequencies <= search_high_hz)
    )

    if not np.any(search_mask):
        return empty_detection_result()

    search_indices = np.where(search_mask)[0]
    relative_peak_indices, _ = find_peaks(psd_db[search_mask])
    candidate_indices = search_indices[relative_peak_indices]

    if candidate_indices.size == 0:
        return empty_detection_result()

    candidate_rows = []

    for peak_index in candidate_indices:
        f0 = float(frequencies[peak_index])

        primary_prominence = local_peak_prominence_db(
            frequencies=frequencies,
            psd_db=psd_db,
            peak_index=peak_index,
            local_background_half_width_hz=(
                detection["local_background_half_width_hz"]
            ),
            peak_exclusion_half_width_hz=(
                detection["peak_exclusion_half_width_hz"]
            ),
        )

        harmonic_2_hz, harmonic_2_prominence = (
            harmonic_prominence_near_target(
                frequencies=frequencies,
                psd_db=psd_db,
                target_hz=2.0 * f0,
                tolerance_hz=detection["harmonic_tolerance_hz"],
                local_background_half_width_hz=(
                    detection["local_background_half_width_hz"]
                ),
                peak_exclusion_half_width_hz=(
                    detection["peak_exclusion_half_width_hz"]
                ),
            )
        )

        harmonic_3_hz, harmonic_3_prominence = (
            harmonic_prominence_near_target(
                frequencies=frequencies,
                psd_db=psd_db,
                target_hz=3.0 * f0,
                tolerance_hz=detection["harmonic_tolerance_hz"],
                local_background_half_width_hz=(
                    detection["local_background_half_width_hz"]
                ),
                peak_exclusion_half_width_hz=(
                    detection["peak_exclusion_half_width_hz"]
                ),
            )
        )

        harmonic_count = int(
            np.isfinite(harmonic_2_prominence)
            and harmonic_2_prominence
            >= detection["minimum_harmonic_prominence_db"]
        ) + int(
            np.isfinite(harmonic_3_prominence)
            and harmonic_3_prominence
            >= detection["minimum_harmonic_prominence_db"]
        )

        score = (
            primary_prominence
            + detection["harmonic_score_weight"] * harmonic_count
        )

        candidate_rows.append(
            {
                "f0_hz": f0,
                "primary_prominence_db": primary_prominence,
                "harmonic_count": harmonic_count,
                "harmonic_2_hz": harmonic_2_hz,
                "harmonic_3_hz": harmonic_3_hz,
                "harmonic_2_prominence_db": harmonic_2_prominence,
                "harmonic_3_prominence_db": harmonic_3_prominence,
                "score": score,
            }
        )

    candidates = pd.DataFrame(candidate_rows)

    accepted = candidates[
        (
            candidates["primary_prominence_db"]
            >= detection["minimum_primary_prominence_db"]
        )
        & (
            candidates["harmonic_count"]
            >= detection["minimum_harmonic_count"]
        )
    ]

    if accepted.empty:
        return empty_detection_result()

    best = (
        accepted.sort_values("score", ascending=False)
        .iloc[0]
        .to_dict()
    )

    best.pop("score")
    best["detected"] = True

    return best


def analyze_recording_windows(
    path: Path,
    condition: str,
    sex: str,
    search_range: tuple[float, float],
    config: dict,
) -> pd.DataFrame:
    audio, sample_rate = read_mono_wav(path)

    window_samples = int(
        config["windowing"]["window_seconds"] * sample_rate
    )
    hop_samples = int(
        config["windowing"]["hop_seconds"] * sample_rate
    )

    if len(audio) < window_samples:
        raise ValueError(
            f"Recording is shorter than one analysis window: {path}"
        )

    rows = []

    for start_sample in range(
        0,
        len(audio) - window_samples + 1,
        hop_samples,
    ):
        end_sample = start_sample + window_samples

        result = estimate_candidate_wbf(
            segment=audio[start_sample:end_sample],
            sample_rate=sample_rate,
            search_low_hz=search_range[0],
            search_high_hz=search_range[1],
            config=config,
        )

        rows.append(
            {
                "condition": condition,
                "sex": sex,
                "file_name": path.name,
                "relative_file_path": str(path),
                "sample_rate_hz": sample_rate,
                "start_s": start_sample / sample_rate,
                "end_s": end_sample / sample_rate,
                "window_duration_s": config["windowing"]["window_seconds"],
                "window_hop_s": config["windowing"]["hop_seconds"],
                "search_low_hz": search_range[0],
                "search_high_hz": search_range[1],
                **result,
            }
        )

    return pd.DataFrame(rows)


def summarize_recordings(windows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        windows.groupby(
            ["condition", "sex", "file_name", "relative_file_path"],
            as_index=False,
        )
        .agg(
            n_windows=("detected", "size"),
            n_detected_windows=("detected", "sum"),
            detection_proportion=("detected", "mean"),
            median_candidate_wbf_hz=("f0_hz", "median"),
            median_peak_prominence_db=(
                "primary_prominence_db",
                "median",
            ),
        )
    )

    summary["detection_percent"] = (
        100.0 * summary["detection_proportion"]
    )

    return summary


def summarize_conditions(recordings: pd.DataFrame) -> pd.DataFrame:
    def q1(values):
        return values.quantile(0.25)

    def q3(values):
        return values.quantile(0.75)

    return (
        recordings.groupby(["condition", "sex"], as_index=False)
        .agg(
            n_recordings=("file_name", "count"),
            median_detection_percent=("detection_percent", "median"),
            q1_detection_percent=("detection_percent", q1),
            q3_detection_percent=("detection_percent", q3),
            minimum_detection_percent=("detection_percent", "min"),
            maximum_detection_percent=("detection_percent", "max"),
            median_recording_wbf_hz=(
                "median_candidate_wbf_hz",
                "median",
            ),
            median_recording_peak_prominence_db=(
                "median_peak_prominence_db",
                "median",
            ),
        )
    )


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def recording_boxplot(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
    groups: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
    ylim=None,
):
    values = [
        dataframe.loc[
            dataframe[group_column] == group,
            value_column,
        ].dropna()
        for group in groups
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
    )

    # Show individual recording-level observations.
    random_generator = np.random.default_rng(2026)

    for position, group_values in enumerate(values, start=1):
        jitter = random_generator.normal(
            loc=position,
            scale=0.025,
            size=len(group_values),
        )

        ax.scatter(
            jitter,
            group_values,
            s=28,
            alpha=0.65,
            color="#1f77b4",
            edgecolor="none",
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FREE-FLIGHT ANALYSIS
# =============================================================================

def analyze_free_flight(
    data_dir: Path,
    output_dir: Path,
    config: dict,
):
    female_directory = (
        data_dir / config["paths"]["free_flying_female"]
    )
    male_directory = (
        data_dir / config["paths"]["free_flying_male"]
    )

    female_files = find_wav_files(female_directory)
    male_files = find_wav_files(male_directory)

    all_windows = []

    for sex, files in [
        ("female", female_files),
        ("male", male_files),
    ]:
        search_range = tuple(
            config["wbf_search_ranges_hz"][sex]
        )

        for number, path in enumerate(files, start=1):
            print(
                f"Free flight, {sex}: "
                f"{number}/{len(files)} — {path.name}"
            )

            all_windows.append(
                analyze_recording_windows(
                    path=path,
                    condition="free_flying",
                    sex=sex,
                    search_range=search_range,
                    config=config,
                )
            )

    windows = pd.concat(all_windows, ignore_index=True)
    recordings = summarize_recordings(windows)
    conditions = summarize_conditions(recordings)

    windows.to_csv(
        output_dir / "free_flight_per_window.csv",
        index=False,
    )
    recordings.to_csv(
        output_dir / "free_flight_per_recording.csv",
        index=False,
    )
    conditions.to_csv(
        output_dir / "free_flight_summary.csv",
        index=False,
    )

    recording_boxplot(
        dataframe=recordings,
        group_column="sex",
        value_column="detection_percent",
        groups=["female", "male"],
        labels=["Female", "Male"],
        ylabel="Candidate-WBF detectable-window proportion (%)",
        title="Free-flying recordings: acoustic candidate-WBF detection",
        output_path=(
            output_dir / "free_flight_detection_by_sex.png"
        ),
        ylim=(0, 100),
    )

    recording_boxplot(
        dataframe=recordings,
        group_column="sex",
        value_column="median_candidate_wbf_hz",
        groups=["female", "male"],
        labels=["Female", "Male"],
        ylabel="Recording-level median candidate WBF (Hz)",
        title="Free-flying recordings: candidate fundamental WBF",
        output_path=(
            output_dir / "free_flight_median_wbf_by_sex.png"
        ),
    )

    recording_boxplot(
        dataframe=recordings,
        group_column="sex",
        value_column="median_peak_prominence_db",
        groups=["female", "male"],
        labels=["Female", "Male"],
        ylabel="Recording-level median WBF peak prominence (dB)",
        title="Free-flying recordings: local WBF peak prominence",
        output_path=(
            output_dir / "free_flight_peak_prominence_by_sex.png"
        ),
    )

    accepted = windows[windows["detected"]].copy()

    for sex in ["female", "male"]:
        values = accepted.loc[
            accepted["sex"] == sex,
            "f0_hz",
        ].dropna()

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(values, bins=60, edgecolor="black")

        ax.set_xlabel("Candidate fundamental WBF (Hz)")
        ax.set_ylabel("Number of accepted 1-s windows")
        ax.set_title(
            f"{sex.capitalize()} free-flying recordings: "
            "accepted candidate fundamental WBF estimates"
        )
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"free_flight_{sex}_wbf_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return windows, recordings, conditions


# =============================================================================
# TETHERED ANALYSIS
# =============================================================================

def analyze_tethered(
    data_dir: Path,
    output_dir: Path,
    config: dict,
):
    outside_directory = (
        data_dir / config["paths"]["tethered_outside_acl2"]
    )
    inside_directory = (
        data_dir / config["paths"]["tethered_inside_acl2_with_enclosure"]
    )

    outside_files = find_wav_files(outside_directory)
    inside_files = find_wav_files(inside_directory)

    all_windows = []
    female_range = tuple(config["wbf_search_ranges_hz"]["female"])

    conditions_and_files = [
        (
            "tethered_outside_acl2_without_enclosure",
            outside_files,
        ),
        (
            "tethered_inside_acl2_with_enclosure",
            inside_files,
        ),
    ]

    for condition, files in conditions_and_files:
        for number, path in enumerate(files, start=1):
            print(
                f"{condition}: {number}/{len(files)} — {path.name}"
            )

            all_windows.append(
                analyze_recording_windows(
                    path=path,
                    condition=condition,
                    sex="female",
                    search_range=female_range,
                    config=config,
                )
            )

    windows = pd.concat(all_windows, ignore_index=True)
    recordings = summarize_recordings(windows)
    conditions = summarize_conditions(recordings)

    windows.to_csv(
        output_dir / "tethered_per_window.csv",
        index=False,
    )
    recordings.to_csv(
        output_dir / "tethered_per_recording.csv",
        index=False,
    )
    conditions.to_csv(
        output_dir / "tethered_summary.csv",
        index=False,
    )

    group_order = [
        "tethered_outside_acl2_without_enclosure",
        "tethered_inside_acl2_with_enclosure",
    ]

    display_labels = [
        "Outside ACL-2,\nwithout enclosure",
        "Inside ACL-2\nenclosure",
    ]

    recording_boxplot(
        dataframe=recordings,
        group_column="condition",
        value_column="detection_percent",
        groups=group_order,
        labels=display_labels,
        ylabel="Candidate-WBF detectable-window proportion (%)",
        title="Female tethered recordings: candidate-WBF detection",
        output_path=output_dir / "tethered_detection.png",
        ylim=(0, 100),
    )

    recording_boxplot(
        dataframe=recordings,
        group_column="condition",
        value_column="median_peak_prominence_db",
        groups=group_order,
        labels=display_labels,
        ylabel="Recording-level median WBF peak prominence (dB)",
        title="Female tethered recordings: WBF peak prominence",
        output_path=output_dir / "tethered_peak_prominence.png",
    )

    return windows, recordings, conditions


# =============================================================================
# AMBIENT ACL-2 VERSUS INSIDE-ENCLOSURE ANALYSIS
# =============================================================================

def common_frequency_grid(
    frequencies_a: np.ndarray,
    frequencies_b: np.ndarray,
) -> np.ndarray:
    low = max(frequencies_a.min(), frequencies_b.min())
    high = min(frequencies_a.max(), frequencies_b.max())

    step_a = np.median(np.diff(frequencies_a))
    step_b = np.median(np.diff(frequencies_b))
    step = max(step_a, step_b)

    return np.arange(low, high + step, step)


def one_third_octave_summary(
    frequencies: np.ndarray,
    outside_psd: np.ndarray,
    inside_psd: np.ndarray,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> pd.DataFrame:
    center_ratio = 2.0 ** (1.0 / 3.0)
    edge_ratio = 2.0 ** (1.0 / 6.0)

    centers = []
    center = minimum_frequency_hz * edge_ratio

    while center / edge_ratio <= maximum_frequency_hz:
        centers.append(center)
        center *= center_ratio

    rows = []

    for center in centers:
        low = max(center / edge_ratio, minimum_frequency_hz)
        high = min(center * edge_ratio, maximum_frequency_hz)

        outside_power = integrate_psd_band(
            frequencies,
            outside_psd,
            low,
            high,
        )

        inside_power = integrate_psd_band(
            frequencies,
            inside_psd,
            low,
            high,
        )

        if (
            np.isfinite(outside_power)
            and np.isfinite(inside_power)
            and outside_power > 0
            and inside_power > 0
        ):
            reduction_db = float(
                power_to_db(outside_power / inside_power)
            )
        else:
            reduction_db = np.nan

        rows.append(
            {
                "center_frequency_hz": center,
                "band_low_hz": low,
                "band_high_hz": high,
                "acl2_integrated_power_linear": outside_power,
                "inside_enclosure_integrated_power_linear": inside_power,
                "ambient_power_reduction_db": reduction_db,
            }
        )

    return pd.DataFrame(rows)


def analyze_ambient(
    data_dir: Path,
    output_dir: Path,
    config: dict,
):
    acl2_path = data_dir / config["paths"]["ambient_acl2"]
    enclosure_path = (
        data_dir / config["paths"]["ambient_inside_enclosure"]
    )

    validate_single_file(acl2_path, "ACL-2 ambient recording")
    validate_single_file(
        enclosure_path,
        "Inside-enclosure ambient recording",
    )

    acl2_audio, acl2_sample_rate = read_mono_wav(acl2_path)
    inside_audio, inside_sample_rate = read_mono_wav(enclosure_path)

    acl2_frequencies, acl2_psd = calculate_psd(
        acl2_audio,
        acl2_sample_rate,
        config["ambient"]["nperseg"],
    )

    inside_frequencies, inside_psd = calculate_psd(
        inside_audio,
        inside_sample_rate,
        config["ambient"]["nperseg"],
    )

    common_frequencies = common_frequency_grid(
        acl2_frequencies,
        inside_frequencies,
    )

    acl2_common = np.interp(
        common_frequencies,
        acl2_frequencies,
        acl2_psd,
    )

    inside_common = np.interp(
        common_frequencies,
        inside_frequencies,
        inside_psd,
    )

    acl2_db = power_to_db(acl2_common)
    inside_db = power_to_db(inside_common)
    binwise_reduction_db = acl2_db - inside_db

    spectrum_table = pd.DataFrame(
        {
            "frequency_hz": common_frequencies,
            "acl2_psd_linear": acl2_common,
            "inside_enclosure_psd_linear": inside_common,
            "acl2_psd_relative_db_per_hz": acl2_db,
            "inside_enclosure_psd_relative_db_per_hz": inside_db,
            "binwise_ambient_power_difference_db": (
                binwise_reduction_db
            ),
        }
    )

    spectrum_table.to_csv(
        output_dir / "ambient_spectrum_comparison.csv",
        index=False,
    )

    band_rows = []

    for band_name, limits in config["ambient"]["analysis_bands_hz"].items():
        low, high = limits

        acl2_power = integrate_psd_band(
            common_frequencies,
            acl2_common,
            low,
            high,
        )

        inside_power = integrate_psd_band(
            common_frequencies,
            inside_common,
            low,
            high,
        )

        reduction_db = (
            float(power_to_db(acl2_power / inside_power))
            if (
                np.isfinite(acl2_power)
                and np.isfinite(inside_power)
                and acl2_power > 0
                and inside_power > 0
            )
            else np.nan
        )

        band_rows.append(
            {
                "band": band_name,
                "band_low_hz": low,
                "band_high_hz": high,
                "acl2_integrated_power_linear": acl2_power,
                "inside_enclosure_integrated_power_linear": inside_power,
                "acl2_band_level_relative_db": (
                    power_to_db(acl2_power)
                ),
                "inside_enclosure_band_level_relative_db": (
                    power_to_db(inside_power)
                ),
                "ambient_power_reduction_db": reduction_db,
            }
        )

    band_summary = pd.DataFrame(band_rows)

    band_summary.to_csv(
        output_dir / "ambient_integrated_band_summary.csv",
        index=False,
    )

    one_third = one_third_octave_summary(
        frequencies=common_frequencies,
        outside_psd=acl2_common,
        inside_psd=inside_common,
        minimum_frequency_hz=config["ambient"]["plot_min_hz"],
        maximum_frequency_hz=config["ambient"]["plot_max_hz"],
    )

    one_third.to_csv(
        output_dir / "ambient_one_third_octave_summary.csv",
        index=False,
    )

    female_low, female_high = (
        config["ambient"]["analysis_bands_hz"]["female_wbf_band"]
    )

    plot_mask = (
        (common_frequencies >= config["ambient"]["plot_min_hz"])
        & (common_frequencies <= config["ambient"]["plot_max_hz"])
    )

    # Ambient spectrum
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    ax.plot(
        common_frequencies[plot_mask],
        acl2_db[plot_mask],
        label="Ambient ACL-2",
        linewidth=1.1,
    )

    ax.plot(
        common_frequencies[plot_mask],
        inside_db[plot_mask],
        label="Ambient inside enclosure",
        linewidth=1.1,
    )

    ax.axvspan(
        female_low,
        female_high,
        color="#5DDDE0",
        alpha=0.28,
        label=f"Female WBF band ({female_low}–{female_high} Hz)",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (relative dBFS/Hz)")
    ax.set_title(
        "Ambient spectrum: ACL-2 versus inside isolation enclosure"
    )
    ax.set_xlim(
        config["ambient"]["plot_min_hz"],
        config["ambient"]["plot_max_hz"],
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "ambient_acl2_vs_inside_spectrum.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    # One-third-octave reduction
    plot_data = one_third.dropna(
        subset=["ambient_power_reduction_db"]
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    ax.plot(
        plot_data["center_frequency_hz"],
        plot_data["ambient_power_reduction_db"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        color="#1B5E20",
        label="1/3-octave band reduction",
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.9,
        linestyle="--",
    )

    ax.axvspan(
        female_low,
        female_high,
        color="#5DDDE0",
        alpha=0.28,
        label=f"Female WBF band ({female_low}–{female_high} Hz)",
    )

    ax.set_xscale("log")
    ax.set_xlim(
        config["ambient"]["plot_min_hz"],
        config["ambient"]["plot_max_hz"],
    )

    ticks = config["ambient"]["one_third_octave_plot_ticks_hz"]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(
        "Observed ambient-power reduction (dB)\n"
        "(ACL-2 ambient − enclosure ambient)"
    )
    ax.set_title(
        "One-third-octave ambient-noise reduction "
        "by the sound-isolation enclosure"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "ambient_noise_reduction_one_third_octave.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    return spectrum_table, band_summary, one_third


# =============================================================================
# SAVE ANALYSIS PROVENANCE
# =============================================================================

def save_analysis_metadata(
    output_dir: Path,
    data_dir: Path,
    config_path: Path,
    config: dict,
):
    metadata = {
        "data_directory": str(data_dir.resolve()),
        "configuration_file": str(config_path.resolve()),
        "analysis_note": (
            "Relative digital spectral measurements; not calibrated dB SPL."
        ),
        "configuration": config,
    }

    with (output_dir / "analysis_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()
    config = load_config(args.config)

    data_dir = args.data_dir.resolve()
    output_directories = make_output_directories(
        args.output_dir.resolve()
    )

    save_analysis_metadata(
        output_dir=output_directories["root"],
        data_dir=data_dir,
        config_path=args.config,
        config=config,
    )

    print("=" * 78)
    print("MOSQUITO WBF ANALYSIS")
    print("=" * 78)
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_directories['root']}")
    print()

    print("1. Analyzing free-flight recordings...")
    free_windows, free_recordings, free_summary = (
        analyze_free_flight(
            data_dir=data_dir,
            output_dir=output_directories["free_flight"],
            config=config,
        )
    )

    print("\n2. Analyzing tethered recordings...")
    tethered_windows, tethered_recordings, tethered_summary = (
        analyze_tethered(
            data_dir=data_dir,
            output_dir=output_directories["tethered"],
            config=config,
        )
    )

    print("\n3. Analyzing matched ACL-2 ambient recordings...")
    ambient_spectrum, ambient_bands, ambient_octaves = (
        analyze_ambient(
            data_dir=data_dir,
            output_dir=output_directories["ambient"],
            config=config,
        )
    )

    print("\nFree-flight summary:")
    print(free_summary.to_string(index=False))

    print("\nTethered descriptive summary:")
    print(tethered_summary.to_string(index=False))

    print("\nAmbient integrated-band summary:")
    print(ambient_bands.to_string(index=False))

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Results were written to: {output_directories['root']}")


if __name__ == "__main__":
    main()