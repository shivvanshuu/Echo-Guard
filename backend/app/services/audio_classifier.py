"""
EchoGuard YAMNet Audio Classification Service
=============================================
Inference-only bioacoustic classifier leveraging Google's pre-trained YAMNet
from TensorFlow Hub (https://tfhub.dev/google/yamnet/1).

Pipeline:
  1. Load .wav file at 16,000 Hz mono (YAMNet standard).
  2. Normalize float32 audio waveform to [-1.0, 1.0].
  3. Run YAMNet inference -> (scores, embeddings, log_mel_spectrogram).
  4. Aggregate frame scores and filter/map 521 AudioSet classes to EchoGuard threats:
     - Chainsaw -> CHAINSAW
     - Gunshot, gunfire / Explosion / Artillery -> GUNSHOT
     - Motorboat / Boat / Watercraft -> ILLEGAL_VESSEL
     - Engine / Motor vehicle / Truck / Car -> VEHICLE_ENGINE
     - Underwater explosion / Blast -> BLAST_FISHING
  5. Compute 128-dimensional unit-normalized acoustic embedding for pgvector.
"""

import os
import sys
import csv
import time
import logging
import platform
import traceback
import threading
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16000  # YAMNet strictly requires 16 kHz
YAMNET_FRAME_SAMPLES = 15360  # 0.96s minimum frame size for YAMNet (0.96 * 16000)

# Domain threat mapping from YAMNet AudioSet classes
# Explosive/firearm classes are grouped so their probabilities are summed collectively for gunshot detection
THREAT_CLASS_GROUPS: Dict[str, List[str]] = {
    # Gunshots / Poaching Firearms / Explosives (Impulsive Threats - summed)
    "GUNSHOT": [
        "Gunshot, gunfire",
        "Cap gun",
        "Explosion",
        "Firecracker",
        "Fireworks",
        "Artillery fire",
        "Machine gun",
        "Fusillade",
        "Boom",
    ],
    # Chainsaw / Deforestation
    "CHAINSAW": [
        "Chainsaw",
    ],
    # Marine / Waterway Sanctuary Breaches
    "ILLEGAL_VESSEL": [
        "Motorboat, speedboat",
        "Boat, Water vehicle",
        "Ship",
        "Watercraft",
    ],
    # Unauthorized Vehicles / Logging Trucks
    "VEHICLE_ENGINE": [
        "Engine",
        "Motor vehicle (road)",
        "Truck",
        "Car",
        "Vehicle",
        "Idling",
        "Accelerating, revving, vroom",
        "Internal combustion engine",
    ],
    # Destructive Fishing
    "BLAST_FISHING": [
        "Burst, pop",
    ],
}

# Flat lookup mapping for backwards compatibility
THREAT_CLASS_MAPPING = {
    class_name: threat_type
    for threat_type, classes in THREAT_CLASS_GROUPS.items()
    for class_name in classes
}

# 521 Standard AudioSet Class Labels for YAMNet
AUDIOSET_CLASSES_SAMPLE = [
    "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
    "Conversation", "Narration, monologue", "Babbling", "Whispering", "Laughter", "Snicker",
    "Chuckle, chortle", "Gasp", "Panting", "Snort", "Cough", "Throat clearing", "Sneezing",
    "Sniff", "Crying, sobbing", "Sigh", "Singing", "Choir", "Yodel", "Chant", "Whispering",
    "Animal", "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Cat",
    "Purr", "Meow", "Hiss", "Caterwaul", "Livestock, farm animals, working animals", "Horse",
    "Clip-clop", "Neigh, whinny", "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink",
    "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo",
    "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk", "Wild animals", "Roaring lions",
    "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Pigeon, dove",
    "Coo", "Crow", "Caw", "Owl", "Hoot", "Bird flight, flapping wings", "Canidae, dogs, wolves",
    "Rodents, rats, mice", "Mouse", "Patter", "Insect", "Cricket", "Mosquito", "Fly, housefly",
    "Buzz", "Bee, wasp, etc.", "Frog", "Croak", "Snake", "Rattle", "Whale vocalization",
    "Music", "Musical instrument", "Plucked string instrument", "Guitar", "Electric guitar",
    "Bass guitar", "Acoustic guitar", "Steel guitar, slide guitar", "Tapping (guitar technique)",
    "Strum", "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele", "Keyboard (musical)", "Piano",
    "Electric piano", "Organ", "Electronic organ", "Hammond organ", "Synthesizer", "Sampler",
    "Harpsichord", "Percussion", "Drum kit", "Snare drum", "Rimshot", "Drum roll", "Bass drum",
    "Timpani", "Tabla", "Cymbal", "Hi-hat", "Crash cymbal", "Cowbell", "Wood block", "Tambourine",
    "Rattle (instrument)", "Maraca", "Gong", "Tubular bells", "Mallet percussion", "Marimba, xylophone",
    "Glockenspiel", "Vibraphone", "Steelpan", "Wind instrument, woodwind instrument", "Flute",
    "Saxophone", "Clarinet", "Oboe", "Bassoon", "Brass instrument", "French horn", "Trumpet",
    "Trombone", "Tuba", "Accordion", "Harmonica", "Bagpipes", "Didgeridoo", "Shofar",
    "Theremin", "Singing bowl", "Scratching (performance technique)", "Pop music", "Hip hop music",
    "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock",
    "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae", "Country",
    "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco", "Classical music",
    "Opera", "Electronic music", "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
    "Electronic dance music", "Ambient music", "Trance music", "Music of Latin America", "Salsa music",
    "Flamenco", "Blues", "Music for children", "New-age music", "Vocal music", "A capella",
    "Music of Africa", "Afrobeat", "Christian music", "Gospel music", "Music of Asia", "Carnatic music",
    "Hindustani music", "Ska", "Traditional music", "Song", "Sound effect", "Echo", "Noise",
    "Environmental noise", "Static", "Mains hum", "Distortion", "Sidetone", "Cacophony", "White noise",
    "Pink noise", "Throbbing", "Vibration", "Television", "Radio", "Field recording", "Chainsaw",
    "Engine", "Motor vehicle (road)", "Car", "Vehicle", "Truck", "Motorcycle", "Idling",
    "Accelerating, revving, vroom", "Internal combustion engine", "Emergency vehicle", "Police car (siren)",
    "Ambulance (siren)", "Fire engine, fire truck (siren)", "Siren", "Civil defense siren",
    "Bicycle", "Skateboard", "Train", "Rail transport", "Train horn", "Train wheels squealing",
    "Subway, metro, underground", "Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew",
    "Helicopter", "Watercraft", "Boat, Water vehicle", "Ship", "Motorboat, speedboat", "Rowboat, canoe, kayak",
    "Sailboat", "Submarine", "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire",
    "Cap gun", "Explosion", "Burst, pop", "Boom", "Thunderstorm", "Thunder", "Water", "Rain",
    "Raindrop", "Stream", "Waterfall", "Ocean", "Waves, surf", "Steam", "Gurgling", "Fire",
    "Crackle", "Wind", "Rustling leaves", "Wind noise (microphone)"
]


THREAT_MIN_CONFIDENCE = 0.45

# Comprehensive AudioSet classes for nature, wildlife, weather, water, and ambient silence
NATURE_AND_SILENCE_CLASSES = {
    # Silence and background noise
    "Silence",
    "Noise",
    "Environmental noise",
    "Static",
    "White noise",
    "Pink noise",
    "Field recording",
    "Outside, rural or natural",
    
    # Natural elements, weather, water
    "Wind",
    "Rustling leaves",
    "Wind noise (microphone)",
    "Thunderstorm",
    "Thunder",
    "Water",
    "Rain",
    "Raindrop",
    "Rain on surface",
    "Stream",
    "Waterfall",
    "Ocean",
    "Waves, surf",
    "Steam",
    "Gurgling",
    "Fire",
    "Crackle",
    
    # Birds and vocalizations
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Squawk",
    "Pigeon, dove",
    "Coo",
    "Crow",
    "Caw",
    "Owl",
    "Hoot",
    "Bird flight, flapping wings",
    
    # Animals, pets, livestock, wildlife
    "Animal",
    "Wild animals",
    "Domestic animals, pets",
    "Dog",
    "Bark",
    "Yip",
    "Howl",
    "Bow-wow",
    "Growling",
    "Whimper (dog)",
    "Cat",
    "Purr",
    "Meow",
    "Hiss",
    "Caterwaul",
    "Livestock, farm animals, working animals",
    "Horse",
    "Clip-clop",
    "Neigh, whinny",
    "Cattle, bovinae",
    "Moo",
    "Pig",
    "Oink",
    "Goat",
    "Bleat",
    "Sheep",
    "Fowl",
    "Chicken, rooster",
    "Cluck",
    "Crowing, cock-a-doodle-doo",
    "Turkey",
    "Gobble",
    "Duck",
    "Quack",
    "Goose",
    "Honk",
    "Roaring cats (lions, tigers)",
    "Roar",
    "Canidae, dogs, wolves",
    "Rodents, rats, mice",
    "Mouse",
    "Patter",
    "Whale vocalization",
    
    # Insects and amphibians
    "Insect",
    "Cricket",
    "Mosquito",
    "Fly, housefly",
    "Buzz",
    "Bee, wasp, etc.",
    "Frog",
    "Croak",
    "Snake",
    "Rattle",
}


def is_nature_or_silence(class_name: str) -> bool:
    """Return True if class_name belongs to nature, wildlife, weather, water, or silence/ambient sound."""
    if not class_name:
        return False
    if class_name in NATURE_AND_SILENCE_CLASSES:
        return True
    lower = class_name.lower()
    keywords = (
        "bird", "animal", "wind", "rain", "silence", "water", "stream", 
        "ocean", "thunder", "insect", "frog", "cricket"
    )
    return any(k in lower for k in keywords) and class_name not in THREAT_CLASS_MAPPING


class YAMNetClassifier:
    """
    TensorFlow Hub YAMNet Model Wrapper.
    """

    # After a failed load, do not try again for this long. Without this, every classification
    # call would wait for the download to time out again.
    RETRY_AFTER_SECONDS = 300

    def __init__(self):
        self._model = None
        self._class_names: List[str] = []
        self._is_loaded = False
        self._lock = threading.Lock()
        self._last_failure = 0.0
        self.load_error: Optional[str] = None

    def _load_model(self):
        if self._is_loaded and self._model is not None:
            return

        with self._lock:
            if self._is_loaded and self._model is not None:
                return
            if self.load_error and (time.monotonic() - self._last_failure) < self.RETRY_AFTER_SECONDS:
                return  # failed recently; stay on the fallback instead of waiting again
            self._load_model_locked()

    def _load_model_locked(self):
        # Base project directories
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Verify candidate paths for YAMNet model weights
        candidate_dirs = [
            os.path.join(base_dir, "ml_models", "yamnet"),
            os.path.join(os.getcwd(), "ml_models", "yamnet"),
            os.path.join(os.getcwd(), "backend", "ml_models", "yamnet"),
        ]
        model_dir = None
        for d in candidate_dirs:
            if os.path.exists(os.path.join(d, "saved_model.pb")):
                model_dir = os.path.abspath(d)
                break
        if not model_dir:
            model_dir = os.path.abspath(os.path.join(base_dir, "ml_models", "yamnet"))

        try:
            # Check and import TensorFlow (handling venv detection)
            try:
                import tensorflow as tf
            except ImportError as tf_err:
                # If running from outside venv, attempt to discover Python 3.11 venv site-packages
                venv_site = os.path.join(base_dir, "venv", "Lib", "site-packages")
                if os.path.exists(venv_site) and venv_site not in sys.path:
                    sys.path.insert(0, venv_site)
                try:
                    import tensorflow as tf
                except ImportError:
                    print(
                        f"\n[ECHO-GUARD CRITICAL] TensorFlow is not available in the active Python environment!\n"
                        f"  Running Python:     {sys.executable} (Python {platform.python_version()})\n"
                        f"  Virtual Env Path:   {os.path.join(base_dir, 'venv')}\n"
                        f"  Exact Error:        {tf_err}\n"
                        f"  Fix: Please launch EchoGuard using the Python 3.11 venv:\n"
                        f"       .\\backend\\venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000\n",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise tf_err

            import requests
            import io
            import tarfile

            # If model weights (saved_model.pb) are missing, download automatically
            saved_model_pb = os.path.join(model_dir, "saved_model.pb")
            if not os.path.exists(model_dir) or not os.path.exists(saved_model_pb):
                msg = f"YAMNet model weights not found at {model_dir}. Downloading from {YAMNET_MODEL_URL}..."
                logger.info(msg)
                print(f"[ECHO-GUARD INFO] {msg}", flush=True)
                os.makedirs(model_dir, exist_ok=True)
                download_url = f"{YAMNET_MODEL_URL}?tf-hub-format=compressed"
                resp = requests.get(download_url, headers={"User-Agent": "EchoGuard-Agent/1.0"}, timeout=60)
                if resp.status_code != 200:
                    raise RuntimeError(f"YAMNet download failed with HTTP {resp.status_code} from {download_url}")
                tar = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
                tar.extractall(model_dir)
                tar.close()
                print(f"[ECHO-GUARD INFO] YAMNet SavedModel extracted successfully to {model_dir}.", flush=True)
                logger.info("YAMNet SavedModel extracted successfully.")

            # Load SavedModel with pure TensorFlow
            logger.info(f"Loading YAMNet SavedModel from {model_dir}...")
            self._model = tf.saved_model.load(model_dir)

            # Load 521 AudioSet class map
            class_map_file = os.path.join(model_dir, "assets", "yamnet_class_map.csv")
            if not os.path.exists(class_map_file):
                class_map_file = self._model.class_map_path().numpy().decode("utf-8") if hasattr(self._model, "class_map_path") else ""

            class_names = []
            if os.path.exists(class_map_file):
                with open(class_map_file, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        class_names.append(row["display_name"])

            if len(class_names) == 0:
                class_names = AUDIOSET_CLASSES_SAMPLE

            self._class_names = class_names
            self._is_loaded = True
            self.load_error = None
            logger.info(f"YAMNet successfully loaded with {len(self._class_names)} classes.")
            print(f"[ECHO-GUARD INFO] Real YAMNet successfully loaded from {model_dir} ({len(self._class_names)} classes).", flush=True)
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            self._last_failure = time.monotonic()
            logger.error(f"Could not load YAMNet SavedModel: {self.load_error}. Using resilient local acoustic model.", exc_info=True)
            print(f"\n[ECHO-GUARD ERROR] Failed to load YAMNet SavedModel: {self.load_error}", file=sys.stderr, flush=True)
            print(f"                 Active Python: {sys.executable} (version {platform.python_version()})", file=sys.stderr, flush=True)
            print(f"                 Model Directory Checked: {model_dir}", file=sys.stderr, flush=True)
            print("                 Falling back to resilient signal heuristic model.\n", file=sys.stderr, flush=True)
            traceback.print_exc()
            self._class_names = AUDIOSET_CLASSES_SAMPLE
            self._is_loaded = False

    def predict(self, waveform: Any) -> Dict[str, Any]:
        """
        Run inference on 16kHz mono audio waveform with peak scoring across 0.96s frames.
        """
        self._load_model()

        if isinstance(waveform, str):
            waveform = load_audio_file(waveform, TARGET_SR)

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Short Audio Padding: Before passing the 16 kHz waveform to YAMNet, check its length.
        # If it is shorter than 0.96 seconds (15,360 samples), pad the array with zeros (silence)
        # to reach the minimum frame size so impulsive sounds are processed instead of returning 0%.
        if len(waveform) < YAMNET_FRAME_SAMPLES:
            waveform = np.pad(
                waveform,
                (0, YAMNET_FRAME_SAMPLES - len(waveform)),
                mode="constant",
                constant_values=0.0,
            )

        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # Check for silence
        rms_val = float(np.sqrt(np.mean(waveform ** 2)))
        is_silence = rms_val < 0.001

        used_real_model = False
        if not is_silence and self._is_loaded and self._model is not None:
            try:
                import tensorflow as tf
                # YAMNet accepts 1D float32 tensor
                waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
                scores, embeddings, log_mel = self._model(waveform_tensor)
                scores_np = scores.numpy()          # [N, 521] - 0.96s frames with 0.48s hop
                embeddings_np = embeddings.numpy()  # [N, 1024]
                log_mel_np = log_mel.numpy()        # [M, 64]

                if scores_np.ndim == 1:
                    scores_np = np.expand_dims(scores_np, axis=0)
                if embeddings_np.ndim == 1:
                    embeddings_np = np.expand_dims(embeddings_np, axis=0)

                used_real_model = True
            except Exception as e:
                logger.warning(f"YAMNet tensor inference failed: {e}. Falling back to DSP heuristic.")
                scores_np, embeddings_np, log_mel_np = self._fallback_dsp_inference(waveform)
        else:
            scores_np, embeddings_np, log_mel_np = self._fallback_dsp_inference(waveform)

        if scores_np.ndim == 1:
            scores_np = np.expand_dims(scores_np, axis=0)
        if embeddings_np.ndim == 1:
            embeddings_np = np.expand_dims(embeddings_np, axis=0)

        num_frames = scores_np.shape[0]

        # Calculate per-threat frame scores using maximum probability among related classes.
        # Avoid summing probabilities which artificially inflates confidence scores up to 100% (1.0).
        category_frame_scores: Dict[str, np.ndarray] = {}
        for threat_type, classes in THREAT_CLASS_GROUPS.items():
            indices = [
                idx for idx, name in enumerate(self._class_names)
                if name in classes and idx < scores_np.shape[1]
            ]
            if indices:
                # Raw peak probability across constituent classes within each 0.96-second frame
                category_frame_scores[threat_type] = np.max(scores_np[:, indices], axis=1)
            else:
                category_frame_scores[threat_type] = np.zeros(num_frames, dtype=np.float32)

        # Peak vs. Average Scoring:
        # Instead of averaging scores across all time frames (which severely dilutes short impulsive
        # sounds like gunshot), extract the maximum peak value (np.max) across the 0.96-second frames.
        matched_threats: Dict[str, float] = {}
        threat_peak_times: Dict[str, float] = {}
        for threat_type, frame_scores in category_frame_scores.items():
            if len(frame_scores) > 0:
                peak_score = float(np.max(frame_scores))
                peak_frame = int(np.argmax(frame_scores))
                matched_threats[threat_type] = min(1.0, peak_score)
                threat_peak_times[threat_type] = round(peak_frame * 0.48, 2)
            else:
                matched_threats[threat_type] = 0.0
                threat_peak_times[threat_type] = 0.0

        # Class probabilities using peak score per class
        peak_scores_per_class = np.max(scores_np, axis=0)
        class_probs: Dict[str, float] = {}
        for idx, class_name in enumerate(self._class_names):
            if idx < len(peak_scores_per_class) and class_name in THREAT_CLASS_MAPPING:
                class_probs[class_name] = round(float(peak_scores_per_class[idx]), 4)

        # Strict Nature Override:
        # Before processing threat labels, find the absolute highest score across all 521 YAMNet classes.
        # In multi-frame audio, inspect the active sound by filtering out leading/trailing silent frames (Silence >= 0.80).
        # If the absolute highest predicted YAMNet class is a nature/background sound (e.g., Bird, Wind, Rain, Silence),
        # immediately force the output to unclassified with 0.0 confidence. Ignore any minor trace scores for vehicles or chainsaws.
        silence_idx = self._class_names.index("Silence") if "Silence" in self._class_names else -1
        if silence_idx >= 0 and scores_np.shape[0] > 1:
            non_silent_mask = scores_np[:, silence_idx] < 0.80
            active_scores = scores_np[non_silent_mask] if np.any(non_silent_mask) else scores_np
        else:
            active_scores = scores_np

        active_peaks = np.max(active_scores, axis=0) if len(active_scores) > 0 else peak_scores_per_class
        abs_top_idx = int(np.argmax(active_peaks)) if len(active_peaks) > 0 else 0
        abs_top_class = self._class_names[abs_top_idx] if abs_top_idx < len(self._class_names) else ""
        abs_top_score = float(active_peaks[abs_top_idx]) if len(active_peaks) > 0 else 0.0

        active_means = np.mean(active_scores, axis=0) if len(active_scores) > 0 else active_peaks
        mean_top_idx = int(np.argmax(active_means)) if len(active_means) > 0 else 0
        mean_top_class = self._class_names[mean_top_idx] if mean_top_idx < len(self._class_names) else ""

        is_nature_discard = is_nature_or_silence(abs_top_class) or is_nature_or_silence(mean_top_class)

        # Disambiguate overlapping vehicle and marine vessel labels.
        # In AudioSet, 'Vehicle' and 'Engine' are parent classes that encompass watercraft,
        # which can cause motorboats to also trigger generic vehicle/engine activations.
        vessel_specific_classes = {
            "Motorboat, speedboat",
            "Boat, Water vehicle",
            "Ship",
            "Watercraft",
        }
        road_vehicle_classes = {
            "Motor vehicle (road)",
            "Truck",
            "Car",
        }
        vessel_peak = max((class_probs.get(c, 0.0) for c in vessel_specific_classes), default=0.0)
        road_peak = max((class_probs.get(c, 0.0) for c in road_vehicle_classes), default=0.0)

        if vessel_peak >= THREAT_MIN_CONFIDENCE and vessel_peak >= road_peak:
            # Marine vessel signature identified; prevent generic engine/vehicle from overriding
            if matched_threats.get("VEHICLE_ENGINE", 0.0) > vessel_peak:
                matched_threats["VEHICLE_ENGINE"] = road_peak
        elif road_peak > vessel_peak and road_peak >= THREAT_MIN_CONFIDENCE:
            # Road vehicle signature identified; prevent low boat noise from overriding
            if matched_threats.get("ILLEGAL_VESSEL", 0.0) > 0.0:
                matched_threats["ILLEGAL_VESSEL"] = vessel_peak

        # 2. Hard Confidence Threshold & Strict Nature Override:
        # If nature/silence override triggers, immediately force the output to unclassified with 0.0 confidence
        # and ignore any minor trace scores for vehicles or chainsaws.
        if is_silence or is_nature_discard:
            threat_type = None
            confidence = 0.0
            peak_time_s = 0.0
            matched_threats = {t: 0.0 for t in matched_threats}
            class_probs = {c: 0.0 for c in class_probs}
        else:
            qualified_threats = {
                t: score for t, score in matched_threats.items()
                if score > THREAT_MIN_CONFIDENCE
            }
            if qualified_threats:
                top_threat = max(qualified_threats.items(), key=lambda x: x[1])
                threat_type = top_threat[0]
                confidence = top_threat[1]
                peak_time_s = threat_peak_times.get(threat_type, 0.0)
            else:
                threat_type = None
                confidence = 0.0
                peak_time_s = 0.0

        # Create 128-dimensional unit-normalized embedding for pgvector compatibility
        # Block-pool 1024-dim down to 128-dim (8-to-1 average pooling)
        mean_embedding = np.mean(embeddings_np, axis=0)
        if len(mean_embedding) == 1024:
            embedding_128 = mean_embedding.reshape(128, 8).mean(axis=1)
        elif len(mean_embedding) >= 128:
            embedding_128 = mean_embedding[:128]
        else:
            embedding_128 = np.pad(mean_embedding, (0, 128 - len(mean_embedding)))

        norm = np.linalg.norm(embedding_128)
        if norm > 0:
            embedding_128 = (embedding_128 / norm).tolist()
        else:
            embedding_128 = embedding_128.tolist()

        # Top five classes across all AudioSet labels by peak score
        top_indices = np.argsort(peak_scores_per_class)[::-1][:5]
        top_classes = [
            {"name": self._class_names[int(i)], "score": round(float(peak_scores_per_class[int(i)]), 4)}
            for i in top_indices
            if int(i) < len(self._class_names)
        ]

        model_name = "Google YAMNet (TensorFlow Hub)" if used_real_model else "Fallback DSP Heuristic (YAMNet unavailable)"

        category_scores = {k: round(float(v), 4) for k, v in matched_threats.items()}

        return {
            "threat_type": threat_type,
            "confidence": round(float(confidence), 4),
            "at_s": peak_time_s,
            "category_scores": category_scores,
            "class_probabilities": class_probs,
            "top_classes": top_classes,
            "embedding": embedding_128,
            "mel_spectrogram": log_mel_np[:32].tolist() if len(log_mel_np) > 0 else [],
            "spectral_analysis": self._compute_spectral_metrics(waveform, TARGET_SR),
            "pipeline": {
                "model": model_name,
                "sample_rate_hz": TARGET_SR,
                "input_format": "16kHz Mono Float32",
                "audioset_classes": len(self._class_names),
                "embedding_dimension": 128,
            },
        }

    def _fallback_dsp_inference(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Acoustic frequency and temporal domain analysis fallback."""
        fft_vals = np.abs(np.fft.rfft(waveform))
        fft_len = len(fft_vals)
        freqs = np.fft.rfftfreq(len(waveform), 1.0 / TARGET_SR)

        # Dominant peak frequency
        dom_idx = int(np.argmax(fft_vals)) if fft_len > 0 else 0
        dom_freq = float(freqs[dom_idx]) if fft_len > 0 else 0.0

        # Energy distribution across bands
        total_energy = float(np.sum(fft_vals) + 1e-9)
        sub_bass = float(np.sum(fft_vals[freqs < 150]) / total_energy)
        low_mid = float(np.sum(fft_vals[(freqs >= 150) & (freqs < 300)]) / total_energy)
        mid_buzz = float(np.sum(fft_vals[(freqs >= 300) & (freqs <= 750)]) / total_energy)
        highs = float(np.sum(fft_vals[freqs > 750]) / total_energy)

        # Transient / Peak factor (Crest Factor)
        peak = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(waveform ** 2)) + 1e-9)
        crest_factor = peak / rms

        scores = np.zeros(len(self._class_names), dtype=np.float32)

        for i, name in enumerate(self._class_names):
            if name == "Chainsaw":
                # Chainsaw has dominant harmonic buzz around 350-600Hz
                if mid_buzz > 0.35 or (300 <= dom_freq <= 650):
                    scores[i] = float(min(0.98, 0.65 + mid_buzz * 0.5))
                else:
                    scores[i] = 0.02
            elif name in ("Gunshot, gunfire", "Explosion", "Cap gun"):
                # Gunshot has sharp impulse with high crest factor (> 4.5)
                if crest_factor > 3.8 and (dom_freq > 800 or highs > 0.3):
                    scores[i] = float(min(0.99, 0.70 + (crest_factor / 10.0) * 0.25))
                else:
                    scores[i] = 0.01
            elif name in ("Engine", "Motor vehicle (road)", "Truck"):
                # Engine has deep low rumble < 150Hz
                if sub_bass > 0.35 or (dom_freq < 150 and crest_factor < 3.5):
                    scores[i] = float(min(0.96, 0.65 + sub_bass * 0.45))
                else:
                    scores[i] = 0.02
            elif name in ("Motorboat, speedboat", "Boat, Water vehicle", "Ship"):
                # Marine vessel has low mid cavitation 150-300Hz
                if low_mid > 0.35 or (150 <= dom_freq < 300):
                    scores[i] = float(min(0.95, 0.60 + low_mid * 0.45))
                else:
                    scores[i] = 0.02
            elif name in ("Burst, pop", "Boom"):
                if crest_factor > 3.0 and dom_freq < 400:
                    scores[i] = float(min(0.92, 0.55 + sub_bass * 0.4))
                else:
                    scores[i] = 0.01
            elif name in ("Noise", "Environmental noise", "Wind", "Rain", "Static"):
                if crest_factor < 3.2 and mid_buzz < 0.35 and sub_bass < 0.35:
                    scores[i] = 0.88

        # 1024-dim synthetic feature embedding from FFT
        if fft_len >= 1024:
            embedding = fft_vals[:1024].astype(np.float32)
        else:
            embedding = np.pad(fft_vals, (0, 1024 - fft_len)).astype(np.float32)

        mel_dummy = np.zeros((32, 64), dtype=np.float32)
        return scores, embedding, mel_dummy

    def _compute_spectral_metrics(self, waveform: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute basic physical bioacoustic metrics."""
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(waveform)))) / 2.0)
        fft_mag = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(len(waveform), 1.0 / sr)
        total_mag = np.sum(fft_mag) + 1e-9
        centroid = float(np.sum(freqs * fft_mag) / total_mag)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / total_mag))

        return {
            "fundamental_frequency_hz": round(float(centroid * 0.25), 2),
            "spectral_centroid_hz": round(centroid, 2),
            "spectral_bandwidth_hz": round(bandwidth, 2),
            "spectral_flatness": round(float(np.exp(np.mean(np.log(fft_mag + 1e-9))) / (np.mean(fft_mag) + 1e-9)), 6),
            "zero_crossing_rate": round(zcr, 6),
            "rms_energy": round(rms, 6),
        }


# Lazy Singleton
_yamnet_instance: Optional[YAMNetClassifier] = None


def get_yamnet_classifier() -> YAMNetClassifier:
    global _yamnet_instance
    if _yamnet_instance is None:
        _yamnet_instance = YAMNetClassifier()
    return _yamnet_instance


def get_classifier():
    """Compatibility interface for legacy callers."""
    return get_yamnet_classifier(), None


def load_audio_file(filepath: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    On-the-fly audio loading and resampling strictly to 16 kHz mono float32.
    Uses librosa.load(path, sr=16000, mono=True) to ensure all incoming audio files
    are strictly converted to 16 kHz mono before passing them to the YAMNet model.
    """
    try:
        import librosa
        waveform, _ = librosa.load(filepath, sr=target_sr, mono=True)
        return waveform.astype(np.float32)
    except Exception as e:
        logger.warning(f"librosa.load failed for {filepath}: {e}, falling back to soundfile/scipy")
        try:
            import soundfile as sf
            data, sr = sf.read(filepath)
            data = np.asarray(data, dtype=np.float32)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                import scipy.signal
                num_samples = int(len(data) * float(target_sr) / sr)
                data = scipy.signal.resample(data, num_samples)
            return data.astype(np.float32)
        except Exception as sf_err:
            raise RuntimeError(f"Could not load and resample audio file {filepath}: {sf_err}")


async def classify_audio_file(filepath: str) -> Dict[str, Any]:
    """
    Asynchronously classify an audio file using YAMNet.
    Ensures strict on-the-fly resampling to 16kHz mono.
    """
    waveform = load_audio_file(filepath, TARGET_SR)
    classifier = get_yamnet_classifier()
    return classifier.predict(waveform)
