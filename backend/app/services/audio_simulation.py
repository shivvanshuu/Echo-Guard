"""
Audio drop simulation
=====================
Takes a real audio or video clip, places it at a point in the forest, works out how
well each sensor node can hear it (distance, loudness, background noise) and classifies
what each node hears with the same classifier the rest of the system uses.

What is real and what is an assumption
--------------------------------------
Real:
  * the clip you upload,
  * the classification of every audio window (YAMNet when it is loaded).

Assumptions (tunable constants below, shown to the user in the response):
  * how loud the clip is at the source,
  * how quickly sound fades with distance,
  * the background noise at each node.
The spatial part is a simulation. It is not a measured field range.
"""
import hashlib
import logging
import math
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.services.clustering import haversine_distance_meters

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MAX_CLIP_SECONDS = 60.0            # only the first minute of a clip is analysed
WINDOW_SECONDS = 2.0               # length of each analysed window
HOP_SECONDS = 1.0                  # step between windows
MIN_WINDOW_RMS = 0.002             # near-silence is skipped (the classifier rescales loudness)
EVENT_WINDOWS_PER_NODE = 3         # strongest moments of the clip each node listens to
ALERT_THRESHOLD = 0.50             # a node raises an alert at or above this confidence

# --- Propagation model & Class-Based Decibel Baselines (assumptions) ---------
REFERENCE_DISTANCE_M = 1.0         # 1 meter reference distance
FOREST_ABSORPTION_DB_PER_KM = 6.0  # extra loss from trees and air per kilometre beyond 1m
NOISE_RMS = 0.01                   # background noise level at every node (about -40 dBFS)
MIN_DETECTION_SNR_DB = 0.0         # a node cannot pick out a sound quieter than its background noise
MAX_SNR_DB = 60.0
LOUDNESS_OFFSET_DB = {"quiet": -10.0, "normal": 0.0, "loud": 10.0}

# Hardcoded real-world baseline source levels (Source Level SPL in dB at 1 meter reference distance)
SOURCE_LEVEL_SPL_DB: Dict[str, float] = {
    "GUNSHOT": 150.0,         # Impulsive high-caliber ballistic impulse (~150 dB SPL @ 1m)
    "BLAST_FISHING": 145.0,    # Explosive detonation shockwave (~145 dB SPL @ 1m)
    "CHAINSAW": 110.0,        # Two-stroke gas logging chainsaw at full throttle (~110 dB SPL @ 1m)
    "ILLEGAL_VESSEL": 105.0,   # High-speed marine outboard vessel engine (~105 dB SPL @ 1m)
    "VEHICLE_ENGINE": 100.0,   # Heavy diesel logging vehicle / road transport truck (~100 dB SPL @ 1m)
}
DEFAULT_SOURCE_LEVEL_SPL_DB = 110.0  # Default baseline source level SPL (~110 dB @ 1m)
AMBIENT_NOISE_FLOOR_SPL_DB = 40.0    # Ambient tropical forest background noise floor (~40 dB SPL, -40 dBFS)

IMPULSIVE_THREATS = {
    "GUNSHOT",
    "BLAST_FISHING",
    "GUNSHOT, GUNFIRE",
    "EXPLOSION",
    "CAP GUN",
    "FIRECRACKER",
    "FIREWORKS",
    "ARTILLERY FIRE",
    "MACHINE GUN",
    "FUSILLADE",
    "BOOM",
    "BURST, POP",
}

SUPPORTED_EXTENSIONS = (
    ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".mp4", ".webm", ".mov", ".mkv",
)


class AudioDecodeError(Exception):
    """The uploaded file could not be turned into audio."""


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    """System ffmpeg if present, otherwise the binary bundled by imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _decode_with_ffmpeg(path: str, ffmpeg: str) -> np.ndarray:
    command = [
        ffmpeg, "-v", "error", "-i", path,
        "-t", str(MAX_CLIP_SECONDS), "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-f", "f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired as error:
        raise AudioDecodeError("Decoding took too long. Try a shorter clip.") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="ignore").strip().splitlines()
        raise AudioDecodeError(f"ffmpeg could not read this file: {detail[-1] if detail else 'unknown error'}")
    if not completed.stdout:
        raise AudioDecodeError("No audio track found in this file.")
    return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)


def _decode_with_soundfile(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sample_rate != TARGET_SR:
        divisor = math.gcd(int(sample_rate), TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // divisor, int(sample_rate) // divisor)
    return mono[: int(MAX_CLIP_SECONDS * TARGET_SR)].astype(np.float32)


def decode_audio(path: str) -> np.ndarray:
    """Return the clip as 16 kHz mono float32, at most MAX_CLIP_SECONDS long."""
    ffmpeg = find_ffmpeg()
    error_text = ""
    if ffmpeg:
        try:
            waveform = _decode_with_ffmpeg(path, ffmpeg)
            if waveform.size:
                return waveform
        except AudioDecodeError as error:
            error_text = str(error)
    try:
        waveform = _decode_with_soundfile(path)
        if waveform.size:
            return waveform
    except Exception as error:  # noqa: BLE001 - any reader failure means "try the next option"
        error_text = error_text or str(error)
    if not ffmpeg:
        raise AudioDecodeError(
            "Could not read this file. Install ffmpeg (or run: pip install imageio-ffmpeg) "
            "to open mp3 and video files."
        )
    raise AudioDecodeError(error_text or "Could not read this file.")


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------
def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0


def window_ranges(n_samples: int) -> List[Tuple[int, int]]:
    """Start and end sample of each analysis window. Short clips get one padded window."""
    size = int(WINDOW_SECONDS * TARGET_SR)
    hop = int(HOP_SECONDS * TARGET_SR)
    if n_samples <= size:
        return [(0, n_samples)]
    return [(start, start + size) for start in range(0, n_samples - size + 1, hop)]


YAMNET_FRAME_SAMPLES = 15360  # 0.96 seconds at 16 kHz minimum YAMNet frame


def pad_to_minimum(window: np.ndarray) -> np.ndarray:
    """Ensure audio window has at least 0.96s (15,360 samples) for YAMNet."""
    if window.size >= YAMNET_FRAME_SAMPLES:
        return window
    return np.pad(window, (0, YAMNET_FRAME_SAMPLES - window.size), mode="constant", constant_values=0.0)


def envelope(waveform: np.ndarray, points: int = 240) -> List[float]:
    """Peak level per slice, 0 to 1, for drawing the waveform."""
    if waveform.size == 0:
        return []
    buckets = np.array_split(np.abs(waveform), min(points, waveform.size))
    peaks = np.array([bucket.max() for bucket in buckets])
    top = float(peaks.max()) or 1.0
    return [round(float(value / top), 3) for value in peaks]


def pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance noise with a 1/f spectrum, closer to wind and rustle than white noise."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if freqs.size > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs / freqs[1] if freqs.size > 1 else 1.0)
    spectrum[0] = 0.0  # no constant offset: real background noise is zero-mean
    noise = np.fft.irfft(spectrum, n_samples)
    std = float(noise.std()) or 1.0
    return (noise / std).astype(np.float32)


def is_impulsive_threat(threat_type: Optional[str]) -> bool:
    """Return True if threat class is an impulsive acoustic event (e.g. gunshots, explosions)."""
    if not threat_type:
        return False
    return threat_type.strip().upper() in IMPULSIVE_THREATS


def get_source_level_spl(threat_type: Optional[str] = None) -> float:
    """Return hardcoded real-world baseline source level (Source Level SPL in dB at 1 meter)."""
    if not threat_type:
        return DEFAULT_SOURCE_LEVEL_SPL_DB
    normalized = threat_type.strip().upper()
    if normalized in SOURCE_LEVEL_SPL_DB:
        return SOURCE_LEVEL_SPL_DB[normalized]
    if is_impulsive_threat(normalized):
        return SOURCE_LEVEL_SPL_DB["GUNSHOT"]
    return DEFAULT_SOURCE_LEVEL_SPL_DB


def calculate_source_intensity(waveform: np.ndarray, threat_type: Optional[str] = None) -> float:
    """
    Calculate the source audio intensity (initial loudness).

    Peak vs. RMS Power:
    If the detected threat is an impulsive class (like gunshot), calculate its initial loudness
    using peak amplitude (max energy) rather than the average RMS power of the entire clip,
    preventing impulsive energy from being diluted across long silence windows.
    For continuous threats (chainsaws, vehicles), average RMS power is used.
    """
    if waveform.size == 0:
        return 0.0
    if is_impulsive_threat(threat_type):
        peak_val = float(np.max(np.abs(waveform)))
        return max(peak_val, 1e-6)
    return rms(waveform)


def received_snr_db(
    distance_m: float,
    loudness: str = "normal",
    threat_type: Optional[str] = None,
) -> float:
    """
    Signal-to-noise ratio at a node in dB.

    Calculates realistic distance drop-off starting from class-based decibel baselines:
      - Geometric spherical spreading loss: -6 dB per doubling of distance (20 * log10(d / 1.0))
      - Atmospheric and forest foliage absorption: extra loss per kilometre
      - Class-based source level baseline: ~150 dB for gunshots, ~110 dB for chainsaws, etc.
    """
    if loudness not in LOUDNESS_OFFSET_DB:
        loudness = "normal"
    distance = max(float(distance_m), 1.0)
    source_spl = get_source_level_spl(threat_type) + LOUDNESS_OFFSET_DB[loudness]

    # Spherical spreading loss (-6 dB per doubling of distance from 1m reference)
    spreading = 20.0 * math.log10(distance)

    # Excess forest foliage and air absorption
    absorption = FOREST_ABSORPTION_DB_PER_KM * max(0.0, distance - 1.0) / 1000.0

    received_spl = source_spl - spreading - absorption
    snr = received_spl - AMBIENT_NOISE_FLOOR_SPL_DB
    return float(min(MAX_SNR_DB, snr))


def detection_radius_m(loudness: str = "normal", threat_type: Optional[str] = None) -> float:
    """Distance at which a node's signal-to-noise ratio falls to the minimum needed to detect."""
    low, high = 1.0, 200000.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if received_snr_db(middle, loudness, threat_type=threat_type) > MIN_DETECTION_SNR_DB:
            low = middle
        else:
            high = middle
    return round(low, 1)


def mix_at_node(
    window: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    threat_type: Optional[str] = None,
) -> np.ndarray:
    """What a node hears: the clip scaled to the right SNR plus background noise."""
    noise = pink_noise(window.size, rng) * NOISE_RMS
    signal_intensity = calculate_source_intensity(window, threat_type)
    if signal_intensity < 1e-9:
        return noise
    scale = NOISE_RMS * (10.0 ** (snr_db / 20.0)) / signal_intensity
    return (window * scale + noise).astype(np.float32)


def _node_seed(base: bytes, node_id: str) -> int:
    return int.from_bytes(hashlib.sha256(base + node_id.encode("utf-8")).digest()[:8], "big")


def _top_classes(result: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
    return (result.get("top_classes") or [])[:count]


def analyze_clip(classifier: Any, waveform: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
    """Classify every window of the clean clip. Returns per-window results and the model name."""
    windows: List[Dict[str, Any]] = []
    model_name = ""
    for start, end in window_ranges(waveform.size):
        chunk = waveform[start:end]
        entry: Dict[str, Any] = {
            "start_s": round(start / TARGET_SR, 2),
            "end_s": round(end / TARGET_SR, 2),
        }
        if rms(chunk) < MIN_WINDOW_RMS:
            entry.update(threat_type="SILENCE", confidence=0.0, top_classes=[])
        else:
            result = classifier.predict(pad_to_minimum(chunk))
            model_name = model_name or result["pipeline"]["model"]
            entry.update(
                threat_type=result["threat_type"],
                confidence=result["confidence"],
                top_classes=_top_classes(result),
            )
        windows.append(entry)
    return windows, model_name


def pick_event_windows(windows: List[Dict[str, Any]], count: int = EVENT_WINDOWS_PER_NODE) -> List[int]:
    """Indices of the strongest threat moments, at least one window length apart."""
    ranked = sorted(
        (i for i, w in enumerate(windows) if w["threat_type"] not in ("SILENCE", "UNKNOWN", "NONE", None)),
        key=lambda i: (windows[i]["threat_type"] not in ("UNKNOWN", "NONE", "SILENCE", None), windows[i]["confidence"]),
        reverse=True,
    )
    chosen: List[int] = []
    for index in ranked:
        if all(abs(windows[index]["start_s"] - windows[c]["start_s"]) >= WINDOW_SECONDS for c in chosen):
            chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def _listen_at_node(
    classifier: Any,
    waveform: np.ndarray,
    windows: List[Dict[str, Any]],
    event_indices: List[int],
    snr_db: float,
    seed: int,
    base_threat: Optional[str] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    ranges = window_ranges(waveform.size)
    best: Optional[Dict[str, Any]] = None
    for index in event_indices:
        start, end = ranges[index]
        window_base_threat = windows[index].get("threat_type")
        target_threat = base_threat if base_threat not in ("UNKNOWN", "NONE", "SILENCE", None) else window_base_threat
        heard = mix_at_node(waveform[start:end], snr_db, rng, threat_type=target_threat)
        result = classifier.predict(pad_to_minimum(heard))

        if target_threat not in ("UNKNOWN", "NONE", "SILENCE", None):
            cat_scores = result.get("category_scores", {})
            target_conf = cat_scores.get(target_threat)
            if target_conf is None:
                if result["threat_type"] == target_threat:
                    target_conf = result["confidence"]
                else:
                    target_conf = result.get("class_probabilities", {}).get(target_threat, 0.0)

            # Verify that the base classifier prediction remains consistent across nodes for a broadcast event,
            # while distance-based audio degradation reduces the raw float confidence score.
            if target_conf >= ALERT_THRESHOLD or result["threat_type"] == target_threat:
                threat_type = target_threat
                conf = target_conf
            else:
                threat_type = None
                conf = target_conf
        else:
            threat_type = result["threat_type"]
            conf = result["confidence"]

        candidate = {
            "threat_type": threat_type,
            "confidence": conf,
            "embedding": result["embedding"],
            "at_s": windows[index]["start_s"],
        }
        is_threat = candidate["threat_type"] not in ("UNKNOWN", "NONE", "SILENCE", None)
        best_is_threat = best is not None and best["threat_type"] not in ("UNKNOWN", "NONE", "SILENCE", None)
        if best is None or (is_threat, candidate["confidence"]) > (best_is_threat, best["confidence"]):
            best = candidate
    return best or {"threat_type": None, "confidence": 0.0, "embedding": None, "at_s": None}


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def simulate_audio_drop(
    classifier: Any,
    waveform: np.ndarray,
    nodes: List[Dict[str, Any]],
    source_lat: float,
    source_lon: float,
    loudness: str = "normal",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns (result, alert_payloads).

    result is JSON-safe and describes the clip, each node and the strongest listener.
    alert_payloads are ready to be stored as alerts, one per node that detected a threat.
    """
    if loudness not in LOUDNESS_OFFSET_DB:
        raise ValueError(f"loudness must be one of {sorted(LOUDNESS_OFFSET_DB)}")

    windows, model_name = analyze_clip(classifier, waveform)
    event_indices = pick_event_windows(windows)
    seed_base = hashlib.sha256(waveform.tobytes()[:200000]).digest()

    threat_windows = [w for w in windows if w["threat_type"] not in ("SILENCE", "UNKNOWN", "NONE", None)]
    peak = max(threat_windows, key=lambda w: w["confidence"]) if threat_windows else None
    source_verdict = {
        "threat_type": peak["threat_type"] if peak else None,
        "confidence": peak["confidence"] if peak else 0.0,
        "at_s": peak["start_s"] if peak else None,
    }

    detected_threat = source_verdict["threat_type"]
    source_intensity = calculate_source_intensity(waveform, detected_threat)
    source_spl = get_source_level_spl(detected_threat)
    source_verdict.update({
        "source_level_spl_db": source_spl,
        "initial_intensity": round(source_intensity, 4),
        "is_impulsive": is_impulsive_threat(detected_threat),
    })

    node_results: List[Dict[str, Any]] = []
    alert_payloads: List[Dict[str, Any]] = []
    for node in nodes:
        distance = haversine_distance_meters(source_lat, source_lon, node["lat"], node["lon"])
        snr = received_snr_db(distance, loudness, threat_type=detected_threat)
        entry: Dict[str, Any] = {
            "id": node["id"],
            "name": node.get("name"),
            "lat": node["lat"],
            "lon": node["lon"],
            "distance_m": round(distance, 1),
            "snr_db": round(snr, 1),
        }
        if snr <= MIN_DETECTION_SNR_DB:
            entry.update(status="too_faint", threat_type=None, confidence=0.0, heard=False, at_s=None)
        elif not event_indices:
            # Calculated SNR is strictly positive (> 0 dB) but no threat detected in the audio clip
            entry.update(status="heard_no_threat", threat_type=None, confidence=0.0, heard=False, at_s=None)
        else:
            listened = _listen_at_node(
                classifier, waveform, windows, event_indices, snr,
                _node_seed(seed_base, node["id"]),
                base_threat=source_verdict["threat_type"],
            )
            heard = listened["threat_type"] not in ("UNKNOWN", "NONE", "SILENCE", None) and listened["confidence"] >= ALERT_THRESHOLD
            entry.update(
                status="detected" if heard else "heard_no_threat",
                threat_type=listened["threat_type"] if heard else None,
                confidence=round(float(listened["confidence"]), 4) if heard else 0.0,
                heard=heard,
                at_s=listened["at_s"] if heard else None,
            )
            if heard:
                alert_payloads.append({
                    "device_id": node["id"],
                    "threat_type": listened["threat_type"],
                    "confidence_score": round(float(listened["confidence"]), 4),
                    "latitude": node["lat"],
                    "longitude": node["lon"],
                    "embedding": listened["embedding"],
                })
        node_results.append(entry)

    node_results.sort(key=lambda n: n["distance_m"])
    detected = [n for n in node_results if n["heard"]]
    strongest = max(detected, key=lambda n: (n["confidence"], -n["distance_m"])) if detected else None

    result = {
        "model": model_name or "unknown",
        "model_is_real_yamnet": model_name.startswith("Google YAMNet"),
        "clip": {
            "duration_s": round(waveform.size / TARGET_SR, 2),
            "envelope": envelope(waveform),
            "windows": windows,
            "verdict_at_source": source_verdict,
        },
        "source": {"lat": source_lat, "lon": source_lon, "loudness": loudness},
        "nodes": node_results,
        "nearest_node": node_results[0]["id"] if node_results else None,
        "strongest_node": strongest["id"] if strongest else None,
        "alert_threshold": ALERT_THRESHOLD,
        "assumptions": {
            "source_level_spl_db": source_spl,
            "ambient_noise_floor_spl_db": AMBIENT_NOISE_FLOOR_SPL_DB,
            "reference_distance_m": REFERENCE_DISTANCE_M,
            "reference_snr_db": source_spl + LOUDNESS_OFFSET_DB[loudness] - AMBIENT_NOISE_FLOOR_SPL_DB,
            "forest_absorption_db_per_km": FOREST_ABSORPTION_DB_PER_KM,
            "minimum_snr_db_to_detect": MIN_DETECTION_SNR_DB,
            "is_impulsive": is_impulsive_threat(detected_threat),
            "note": "Classification is real. Distance, loudness and noise are simulated assumptions.",
        },
    }
    return result, alert_payloads
