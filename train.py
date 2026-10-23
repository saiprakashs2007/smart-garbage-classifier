from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# The script searches common folder layouts and finds the folder containing
# the six class folders, so your dataset can be nested one level deeper.
EXPECTED_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}

def find_dataset_dir():
    candidates = [
        PROJECT_DIR / "dataset" / "dataset-resized",
        PROJECT_DIR / "dataset",
        PROJECT_DIR / "dataset-resized",
    ]
    candidates += [p for p in (PROJECT_DIR / "dataset").glob("**/*") if p.is_dir()] if (PROJECT_DIR / "dataset").exists() else []
    seen = set()
    for folder in candidates:
        try:
            folder = folder.resolve()
        except OSError:
            continue
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        subdirs = {p.name.lower() for p in folder.iterdir() if p.is_dir()}
        if EXPECTED_CLASSES.issubset(subdirs):
            return folder
    raise FileNotFoundError(
        "Could not find the six category folders. Expected cardboard, glass, "
        "metal, paper, plastic and trash inside a folder under 'dataset'."
    )

def main():
    tf.keras.utils.set_random_seed(42)
    data_dir = find_dataset_dir()
    print(f"\nUsing dataset: {data_dir}")

    image_size = (224, 224)
    batch_size = 32

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    (MODEL_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    # MobileNetV2 is lightweight and is a practical starting point for later Pi deployment.
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "garbage_classifier.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\nTraining classifier head. First run may download MobileNetV2 weights.")
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {accuracy:.1%}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'garbage_classifier.keras'}")
    print(f"Class names saved to: {MODEL_DIR / 'class_names.json'}")
    print("\nNote: validation accuracy is an estimate, not a guarantee of real-world performance.")

if __name__ == "__main__":
    main()
