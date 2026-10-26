import io
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.services import audio_simulation as sim
from app.services.audio_classifier import AUDIOSET_CLASSES_SAMPLE, YAMNetClassifier

SR = sim.TARGET_SR


@pytest.fixture
def fallback_classifier(monkeypatch):
    """A classifier locked to the signal heuristic, so results do not depend on TensorFlow."""
    classifier = YAMNetClassifier()
    monkeypatch.setattr(classifier, "_load_model", lambda: None)
    classifier._class_names = AUDIOSET_CLASSES_SAMPLE
    classifier._is_loaded = False
    return classifier


def chainsaw_like(seconds=6.0, sr=SR):
    """Harmonic buzz around 450 Hz, the kind of signal the heuristic reads as a chainsaw."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = sum(np.sin(2 * np.pi * 450 * k * t) / k for k in (1, 2, 3))
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 7 * t)
    return (wave / np.max(np.abs(wave)) * 0.6).astype(np.float32)


NODES = [
    {"id": "NEAR", "lat": -3.4653, "lon": -62.2159, "name": "Near node"},
    {"id": "MID", "lat": -3.4653, "lon": -62.2159 + 0.0050, "name": "Mid node"},   # about 550 m east
    {"id": "FAR", "lat": -3.4653, "lon": -62.2159 + 0.0450, "name": "Far node"},   # about 5 km east
]


# ---- propagation model ------------------------------------------------------
def test_snr_falls_with_distance():
    values = [sim.received_snr_db(d, "normal") for d in (10, 100, 500, 1000, 3000)]
    assert values == sorted(values, reverse=True)


def test_doubling_distance_costs_about_six_db_near_the_source():
    near = sim.received_snr_db(200, "normal")
    farther = sim.received_snr_db(400, "normal")
    assert 5.5 < near - farther < 7.5


def test_louder_source_is_heard_further():
    assert sim.received_snr_db(800, "loud") > sim.received_snr_db(800, "normal") > sim.received_snr_db(800, "quiet")


def test_mix_hits_requested_snr():
    rng = np.random.default_rng(1)
    clip = chainsaw_like(2.0)
    mixed = sim.mix_at_node(clip, 20.0, rng)
    noise_only = sim.mix_at_node(np.zeros_like(clip), 20.0, np.random.default_rng(1))
    signal_part = mixed - noise_only
    measured = 20 * np.log10(sim.rms(signal_part) / sim.rms(noise_only))
    assert abs(measured - 20.0) < 0.5


# ---- decoding ---------------------------------------------------------------
def test_decode_resamples_and_downmixes(tmp_path):
    sr = 44100
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 300 * t)], axis=1).astype(np.float32) * 0.5
    path = tmp_path / "clip.wav"
    sf.write(str(path), stereo, sr)
    decoded = sim.decode_audio(str(path))
    assert decoded.ndim == 1
    assert abs(decoded.size - int(1.5 * SR)) < SR * 0.05


@pytest.mark.skipif(sim.find_ffmpeg() is None, reason="ffmpeg not available")
def test_decode_mp3_and_mp4(tmp_path):
    ffmpeg = sim.find_ffmpeg()
    wav = tmp_path / "in.wav"
    sf.write(str(wav), chainsaw_like(3.0), SR)

    mp3 = tmp_path / "clip.mp3"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    decoded_mp3 = sim.decode_audio(str(mp3))
    assert abs(decoded_mp3.size / SR - 3.0) < 0.3

    mp4 = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-i", str(wav),
         "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(mp4)],
        check=True,
    )
    decoded_mp4 = sim.decode_audio(str(mp4))
    assert abs(decoded_mp4.size / SR - 3.0) < 0.3
    assert sim.rms(decoded_mp4) > 0.05


def test_decode_rejects_garbage(tmp_path):
    path = tmp_path / "bad.mp3"
    path.write_bytes(b"this is not audio at all")
    with pytest.raises(sim.AudioDecodeError):
        sim.decode_audio(str(path))


# ---- simulation -------------------------------------------------------------
def test_nearest_node_detects_with_highest_confidence(fallback_classifier):
    result, payloads = sim.simulate_audio_drop(
        fallback_classifier, chainsaw_like(), NODES, -3.4653, -62.2159 - 0.0003, "normal"
    )
    by_id = {n["id"]: n for n in result["nodes"]}
    assert result["nearest_node"] == "NEAR"
    assert by_id["NEAR"]["heard"] is True
    assert by_id["NEAR"]["threat_type"] == "CHAINSAW"
    assert by_id["FAR"]["heard"] is False
    assert by_id["FAR"]["status"] == "too_faint"
    assert result["strongest_node"] == "NEAR"
    assert by_id["NEAR"]["confidence"] >= by_id["MID"]["confidence"]


def test_alert_payloads_match_detections(fallback_classifier):
    result, payloads = sim.simulate_audio_drop(
        fallback_classifier, chainsaw_like(), NODES, -3.4653, -62.2159, "normal"
    )
    detected = {n["id"] for n in result["nodes"] if n["heard"]}
    assert {p["device_id"] for p in payloads} == detected
    for payload in payloads:
        assert payload["threat_type"] != "UNKNOWN"
        assert 0.0 <= payload["confidence_score"] <= 1.0
        assert len(payload["embedding"]) == 128


def test_same_inputs_give_same_result(fallback_classifier):
    args = (fallback_classifier, chainsaw_like(), NODES, -3.4653, -62.2150, "normal")
    first, _ = sim.simulate_audio_drop(*args)
    second, _ = sim.simulate_audio_drop(*args)
    assert [n["confidence"] for n in first["nodes"]] == [n["confidence"] for n in second["nodes"]]


def test_louder_source_reaches_more_nodes(fallback_classifier):
    quiet, _ = sim.simulate_audio_drop(fallback_classifier, chainsaw_like(), NODES, -3.4653, -62.2159, "quiet")
    loud, _ = sim.simulate_audio_drop(fallback_classifier, chainsaw_like(), NODES, -3.4653, -62.2159, "loud")
    assert sum(n["status"] != "too_faint" for n in loud["nodes"]) >= sum(n["status"] != "too_faint" for n in quiet["nodes"])


def test_silent_clip_raises_nothing(fallback_classifier):
    silence = np.zeros(SR * 4, dtype=np.float32)
    result, payloads = sim.simulate_audio_drop(fallback_classifier, silence, NODES, -3.4653, -62.2159, "normal")
    assert payloads == []
    assert result["strongest_node"] is None
    assert all(w["threat_type"] == "SILENCE" for w in result["clip"]["windows"])


def test_positive_snr_unclassified_sound_labels_heard_no_threat(fallback_classifier):
    """When SNR > 0 dB but sound is unclassified (0%), status must be heard_no_threat, not too_faint."""
    t = np.linspace(0, 4.0, int(SR * 4.0), endpoint=False)
    nature_tone = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    result, payloads = sim.simulate_audio_drop(
        fallback_classifier, nature_tone, NODES, -3.4653, -62.2159, "normal"
    )
    by_id = {n["id"]: n for n in result["nodes"]}
    near = by_id["NEAR"]
    assert near["snr_db"] > 0
    assert near["confidence"] == 0.0
    assert near["heard"] is False
    assert near["status"] == "heard_no_threat"

    far = by_id["FAR"]
    assert far["snr_db"] <= 0
    assert far["status"] == "too_faint"



def test_result_reports_which_model_ran(fallback_classifier):
    result, _ = sim.simulate_audio_drop(fallback_classifier, chainsaw_like(), NODES, -3.4653, -62.2159, "normal")
    assert result["model_is_real_yamnet"] is False
    assert "Fallback" in result["model"]
    assert len(result["clip"]["envelope"]) > 10
    assert result["clip"]["windows"][0]["top_classes"] is not None


def test_invalid_loudness_is_rejected(fallback_classifier):
    with pytest.raises(ValueError):
        sim.simulate_audio_drop(fallback_classifier, chainsaw_like(), NODES, 0.0, 0.0, "deafening")


def test_node_below_noise_floor_never_alerts(fallback_classifier):
    """Even if the classifier would guess something from noise, a node under the noise floor stays silent."""
    result, payloads = sim.simulate_audio_drop(
        fallback_classifier, chainsaw_like(), [NODES[2]], -3.4653, -62.2159, "normal"
    )
    node = result["nodes"][0]
    assert node["snr_db"] < sim.MIN_DETECTION_SNR_DB
    assert node["status"] == "too_faint" and node["confidence"] == 0.0 and not node["heard"]
    assert payloads == []


def test_detection_radius_matches_the_noise_floor_rule():
    for loudness in ("quiet", "normal", "loud"):
        radius = sim.detection_radius_m(loudness)
        assert abs(sim.received_snr_db(radius, loudness) - sim.MIN_DETECTION_SNR_DB) < 0.05
    assert sim.detection_radius_m("quiet") < sim.detection_radius_m("normal") < sim.detection_radius_m("loud")


def test_broadcast_event_prediction_consistency_across_nodes(fallback_classifier):
    """
    Verify that when a single sound event is broadcast to multiple nodes,
    detecting nodes consistently predict the base event threat type and
    confidence scores reflect distance-based degradation without being pegged to 1.0.
    """
    multi_nodes = [
        {"id": "NODE_CLOSE_1", "lat": -3.4653, "lon": -62.2159, "name": "Node Close 1"},
        {"id": "NODE_CLOSE_2", "lat": -3.4654, "lon": -62.2158, "name": "Node Close 2"},
        {"id": "NODE_MID", "lat": -3.4653, "lon": -62.2130, "name": "Node Mid"},
    ]
    result, payloads = sim.simulate_audio_drop(
        fallback_classifier, chainsaw_like(), multi_nodes, -3.4653, -62.2159, "normal"
    )
    detected = [n for n in result["nodes"] if n["heard"]]
    assert len(detected) >= 2
    # Base classifier prediction consistency
    threat_types = {n["threat_type"] for n in detected}
    assert len(threat_types) == 1
    assert "CHAINSAW" in threat_types
    # Raw float probabilities, not hardcoded to 1.0
    for n in detected:
        assert isinstance(n["confidence"], float)
        assert 0.0 < n["confidence"] <= 1.0


def test_peak_vs_rms_power_for_impulsive_threats():
    """Verify that impulsive threats calculate initial intensity using peak amplitude rather than RMS."""
    # 2-second clip with a 20ms transient impulse
    duration_samples = SR * 2
    clip = np.zeros(duration_samples, dtype=np.float32)
    impulse_len = int(SR * 0.02)  # 20ms
    clip[:impulse_len] = 0.95  # Peak 0.95

    # Peak amplitude for impulsive gunshot
    gunshot_intensity = sim.calculate_source_intensity(clip, "GUNSHOT")
    assert gunshot_intensity == pytest.approx(0.95, abs=1e-3)

    # Average RMS power for continuous chainsaw
    chainsaw_intensity = sim.calculate_source_intensity(clip, "CHAINSAW")
    clip_rms = sim.rms(clip)
    assert chainsaw_intensity == pytest.approx(clip_rms, abs=1e-3)
    assert gunshot_intensity > chainsaw_intensity * 8.0

    assert sim.is_impulsive_threat("GUNSHOT") is True
    assert sim.is_impulsive_threat("BLAST_FISHING") is True
    assert sim.is_impulsive_threat("CHAINSAW") is False
    assert sim.is_impulsive_threat("VEHICLE_ENGINE") is False


def test_class_based_decibel_baselines():
    """Verify hardcoded real-world baseline source levels (Source Level SPL in dB at 1m)."""
    assert sim.SOURCE_LEVEL_SPL_DB["GUNSHOT"] == 150.0
    assert sim.SOURCE_LEVEL_SPL_DB["BLAST_FISHING"] == 145.0
    assert sim.SOURCE_LEVEL_SPL_DB["CHAINSAW"] == 110.0
    assert sim.SOURCE_LEVEL_SPL_DB["ILLEGAL_VESSEL"] == 105.0
    assert sim.SOURCE_LEVEL_SPL_DB["VEHICLE_ENGINE"] == 100.0


def test_gunshot_propagates_further_than_chainsaw_and_vehicle():
    """Verify realistic distance drop-off allows gunshots to propagate much further than engines or chainsaws."""
    gunshot_radius = sim.detection_radius_m("normal", threat_type="GUNSHOT")
    chainsaw_radius = sim.detection_radius_m("normal", threat_type="CHAINSAW")
    vehicle_radius = sim.detection_radius_m("normal", threat_type="VEHICLE_ENGINE")

    # Gunshots propagate multiple kilometres across the mesh (~5.5 km)
    assert gunshot_radius > 4000.0
    assert chainsaw_radius < 1500.0
    assert vehicle_radius < 800.0
    assert gunshot_radius > chainsaw_radius * 3.5

    # At 1.5 km: gunshot is still loud and clear (> 35 dB SNR), while chainsaw and vehicle drop below noise floor
    dist_1_5km = 1500.0
    assert sim.received_snr_db(dist_1_5km, "normal", threat_type="GUNSHOT") > 35.0
    assert sim.received_snr_db(dist_1_5km, "normal", threat_type="CHAINSAW") < sim.MIN_DETECTION_SNR_DB
    assert sim.received_snr_db(dist_1_5km, "normal", threat_type="VEHICLE_ENGINE") < sim.MIN_DETECTION_SNR_DB


