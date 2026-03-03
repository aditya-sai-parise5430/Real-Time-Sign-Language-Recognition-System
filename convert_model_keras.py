"""
Convert Keras model (JSON + H5 weights) to a single portable .keras file.
This bypasses the `tensorflowjs` pip install bug on Windows.
Once this script creates `portable_model.keras`, we can load it directly or convert it via a Colab snippet.

Run:
    python convert_model_keras.py
"""

import json
from pathlib import Path
import numpy as np
from tensorflow.keras.models import model_from_json, save_model  # type: ignore

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

# Save as modern .keras format (easier to handle than split json+h5)
out_path = ROOT / "portable_model.keras"
save_model(model, str(out_path))
print(f"✅ Portable Keras model saved: {out_path}")

print("\n--- NEXT STEPS ---")
print("Because `tensorflowjs` fails to install on Windows, you have two options:")
print("1) Use Google Colab (Free, 1 minute) to convert `portable_model.keras` to TF.js format.")
print("2) Let the browser app load the H5 model directly (not officially supported by TF.js, but sometimes works via converters).")
