import os
import pytest
import numpy as np
import tempfile
import soundfile as sf
from app.services.audio_classifier import (
    classify_audio_file,
    get_yamnet_classifier,
    TARGET_SR,
    THREAT_CLASS_MAPPING,
    THREAT_MIN_CONFIDENCE,
    is_nature_or_silence,
)

@pytest.mark.asyncio
async def test_yamnet_classifier_initialization():
    classifier = get_yamnet_classifier()
    assert classifier is not None
    classifier._load_model()
    assert len(classifier._class_names) > 0


@pytest.mark.asyncio
async def test_yamnet_inference_waveform():
    classifier = get_yamnet_classifier()
    # 1.5s synthetic waveform at 16kHz
    duration = 1.5
    t = np.linspace(0, duration, int(TARGET_SR * duration), endpoint=False)
    # 450Hz tone with harmonics (chainsaw-like)
    waveform = 0.6 * np.sin(2 * np.pi * 450 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)

    result = classifier.predict(waveform)
    assert "threat_type" in result
    assert "confidence" in result
    assert "embedding" in result
    assert len(result["embedding"]) == 128  # Must match pgvector 128-dim schema
    assert "spectral_analysis" in result
    assert result["pipeline"]["sample_rate_hz"] == 16000


@pytest.mark.asyncio
async def test_classify_audio_file_wav():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        t = np.linspace(0, 1.0, TARGET_SR, endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * 1200 * t)
        sf.write(wav_path, waveform.astype(np.float32), TARGET_SR)

        result = await classify_audio_file(wav_path)
        assert "threat_type" in result
        assert "confidence" in result
        assert "embedding" in result
        assert len(result["embedding"]) == 128
        assert "spectral_analysis" in result
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_confidence_not_hardcoded_or_rounded_to_1():
    """Verify raw float probability from YAMNet is returned and not artificially summed or clamped to 1.0."""
    classifier = get_yamnet_classifier()
    # Create multi-frequency tone
    t = np.linspace(0, 1.5, TARGET_SR * 2, endpoint=False)
    waveform = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    result = classifier.predict(waveform)
    
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    # Unless model is 100% theoretically certain, raw probability must not be artificially pegged to 1.0
    assert result["confidence"] < 1.0 or result["threat_type"] == "UNKNOWN"
    assert "category_scores" in result


def test_overlapping_vehicle_and_vessel_disambiguation():
    """Verify that specific vessel classes disambiguate from generic vehicle/engine classes."""
    classifier = get_yamnet_classifier()
    # Check if real clips are present for regression
    vessel_clip = os.path.join("clips", "illegal_vessel", "8q-RUwRrn-U.mp3")
    vehicle_clip = os.path.join("clips", "vehicle", "9i2tNw90RbQ.mp3")

    if os.path.exists(vessel_clip) and classifier._is_loaded:
        vessel_result = classifier.predict(vessel_clip)
        assert vessel_result["threat_type"] == "ILLEGAL_VESSEL"
        assert 0.0 < vessel_result["confidence"] < 1.0

    if os.path.exists(vehicle_clip) and classifier._is_loaded:
        vehicle_result = classifier.predict(vehicle_clip)
        assert vehicle_result["threat_type"] == "VEHICLE_ENGINE"
        assert 0.0 < vehicle_result["confidence"] < 1.0


def test_is_nature_or_silence_helper():
    """Verify nature and silence classes correctly identify without threat false positives."""
    assert is_nature_or_silence("Bird")
    assert is_nature_or_silence("Bird vocalization, bird call, bird song")
    assert is_nature_or_silence("Animal")
    assert is_nature_or_silence("Wind")
    assert is_nature_or_silence("Rain")
    assert is_nature_or_silence("Silence")
    assert is_nature_or_silence("Stream")
    assert is_nature_or_silence("Cricket")
    
    # Threats must NEVER be classified as nature or silence
    assert not is_nature_or_silence("Chainsaw")
    assert not is_nature_or_silence("Gunshot, gunfire")
    assert not is_nature_or_silence("Vehicle")
    assert not is_nature_or_silence("Motorboat, speedboat")


def test_global_top_class_nature_silence_discard():
    """Verify that when top predicted class is nature or silence, threat detection is discarded (None)."""
    classifier = get_yamnet_classifier()
    none_dir = os.path.join("clips", "none")
    if os.path.exists(none_dir) and classifier._is_loaded:
        for file in os.listdir(none_dir):
            if file.endswith(".mp3") or file.endswith(".wav"):
                clip_path = os.path.join(none_dir, file)
                result = classifier.predict(clip_path)
                assert result["threat_type"] is None
                assert result["confidence"] == 0.0


def test_hard_confidence_threshold_safeguard():
    """Verify that threats <= 0.45 confidence default to None and do not trigger alerts."""
    classifier = get_yamnet_classifier()
    assert THREAT_MIN_CONFIDENCE == 0.45

    # Pure silence waveform: all threat scores will be low
    silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
    result = classifier.predict(silence)
    assert result["threat_type"] is None
    assert result["confidence"] == 0.0

    # Low amplitude arbitrary noise below 0.45 threshold
    np.random.seed(42)
    gentle_noise = (np.random.randn(TARGET_SR * 2) * 0.005).astype(np.float32)
    noise_result = classifier.predict(gentle_noise)
    assert noise_result["threat_type"] is None
    assert noise_result["confidence"] == 0.0


def test_short_audio_padding_to_minimum_frame():
    """Verify that waveforms shorter than 0.96s (15,360 samples) are padded with zeros and processed."""
    classifier = get_yamnet_classifier()
    # 0.25 second impulse (4,000 samples < 15,360 samples)
    t = np.linspace(0, 0.25, int(TARGET_SR * 0.25), endpoint=False)
    short_waveform = (0.8 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)
    assert len(short_waveform) < 15360

    result = classifier.predict(short_waveform)
    assert "threat_type" in result
    assert "confidence" in result
    assert "embedding" in result
    assert len(result["embedding"]) == 128
    assert "spectral_analysis" in result


def test_strict_nature_override_ignores_trace_threats():
    """Verify that if the highest predicted class is nature/background, output is unclassified (0.0 confidence)."""
    classifier = get_yamnet_classifier()
    # Mock real model inference or simulate scores where Bird is top class with minor chainsaw trace
    bird_idx = classifier._class_names.index("Bird") if "Bird" in classifier._class_names else 0
    chainsaw_idx = classifier._class_names.index("Chainsaw") if "Chainsaw" in classifier._class_names else 1

    num_classes = len(classifier._class_names)
    synthetic_scores = np.zeros((2, num_classes), dtype=np.float32)
    synthetic_scores[:, bird_idx] = 0.85  # Highest predicted class: Nature
    synthetic_scores[:, chainsaw_idx] = 0.20  # Minor trace score

    class MockModel:
        def __call__(self, tensor):
            import tensorflow as tf
            scores_t = tf.convert_to_tensor(synthetic_scores, dtype=tf.float32)
            embeddings_t = tf.zeros((2, 1024), dtype=tf.float32)
            mel_t = tf.zeros((32, 64), dtype=tf.float32)
            return scores_t, embeddings_t, mel_t

    orig_model = classifier._model
    orig_loaded = classifier._is_loaded
    try:
        classifier._model = MockModel()
        classifier._is_loaded = True

        waveform = np.ones(TARGET_SR, dtype=np.float32) * 0.1
        result = classifier.predict(waveform)

        assert result["threat_type"] is None
        assert result["confidence"] == 0.0
        # Ensure trace threat scores are ignored / zeroed
        assert result["category_scores"].get("CHAINSAW", 0.0) == 0.0
    finally:
        classifier._model = orig_model
        classifier._is_loaded = orig_loaded


