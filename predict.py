from pathlib import Path
import json
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "garbage_classifier.keras"
CLASSES_PATH = PROJECT_DIR / "models" / "class_names.json"

def predict(image_path):
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Model files not found. Run 'python train.py' first.")
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    image = Image.open(path).convert("RGB").resize((224, 224))
    pixels = np.asarray(image, dtype=np.float32)
    probabilities = model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    print(f"\nImage: {path.name}")
    print(f"Predicted category: {class_names[best_index]}")
    print(f"Model confidence: {probabilities[best_index]:.1%}")
    print("\nAll class scores:")
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_names[int(i)]:10s} {probabilities[int(i)]:.1%}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "path\\to\\image.jpg"')
        raise SystemExit(1)
    predict(sys.argv[1])
