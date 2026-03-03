"""
Convert Keras model (JSON + H5 weights) → TensorFlow.js LayersModel
Output goes to docs/tfjs_model/ for GitHub Pages deployment.

Run:
    python convert_model.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
from tensorflow.keras.models import model_from_json  # type: ignore
import tensorflowjs as tfjs  # pip install tensorflowjs

# ──────────────────────────────────────────
# 1. Load the best available model
# ──────────────────────────────────────────
ROOT = Path(__file__).parent

pairs = [
    ("model33k.json",       "newmodel33k.h5"),
    ("model(0.35).json",    "newmodel(0.35).h5"),
    ("model(0.2).json",     "newmodel(0.2).h5"),
    ("model_improved.json", "newmodel_improved.h5"),
]

model = None
used_pair = None
for j, h in pairs:
    jp, hp = ROOT / j, ROOT / h
    if jp.exists() and hp.exists():
        print(f"Loading: {j} + {h}")
        with open(jp) as f:
            model = model_from_json(f.read())
        model.load_weights(str(hp))
        used_pair = (j, h)
        print("✅ Model loaded successfully")
        break

if model is None:
    raise RuntimeError("❌ No model pair found in project root!")

# Quick sanity check
dummy = np.zeros((1, 30, 63))
pred = model.predict(dummy, verbose=0)
print(f"✅ Model outputs shape: {pred.shape} — classes: {pred.shape[-1]}")

# ──────────────────────────────────────────
# 2. Save as complete .h5 (full model)
# ──────────────────────────────────────────
tmp_h5 = ROOT / "tmp_full_model.h5"
model.save(str(tmp_h5))
print(f"✅ Temporary full model saved: {tmp_h5}")

# ──────────────────────────────────────────
# 3. Convert to TF.js LayersModel
# ──────────────────────────────────────────
OUT_DIR = ROOT / "docs" / "tfjs_model"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

tfjs.converters.save_keras_model(model, str(OUT_DIR))
print(f"✅ TF.js model saved to: {OUT_DIR}")

# ──────────────────────────────────────────
# 4. Save model metadata for the frontend
# ──────────────────────────────────────────
meta = {
    "source_files": list(used_pair),
    "input_shape": [30, 63],
    "num_classes": 26,
    "actions": [chr(i) for i in range(65, 91)],
    "threshold": 0.82,
    "sequence_length": 30,
    "consistency_frames": 8,
}
meta_path = ROOT / "docs" / "tfjs_model" / "model_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ Metadata saved: {meta_path}")

# ──────────────────────────────────────────
# 5. Cleanup temp file
# ──────────────────────────────────────────
tmp_h5.unlink()
print("\n🎉 Conversion complete! Files in: docs/tfjs_model/")
print("    Next: Open docs/index.html in your browser.")
