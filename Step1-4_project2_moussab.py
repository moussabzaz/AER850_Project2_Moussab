"""
AER850 Project 2 — Steps 1–4 
Author: Moussab Arfat Zaz — 501082410
"""

# ——— STEP 1: DATA PIPELINE ———
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Core knobs
IMG_H, IMG_W = 500, 500
BATCH_SZ = 32
INPUT_SHAPE = (IMG_H, IMG_W, 3)
NUM_CLASSES = 3
EPOCHS = 25
LR = 1e-4

#  Folder Path
base_dir   = Path(r"C:/Users/mouss/OneDrive/Desktop/Data")
base_train = base_dir / "Train"
base_valid = base_dir / "Valid"
base_test  = base_dir / "Test"   # optional

#Quick existence check
for p in [base_train, base_valid, base_test]:
    if not p.exists():
        print(f"⚠️ Folder not found: {p}")

# Image generators
train_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
)

valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)

# Directory iterators
train_ds = train_gen.flow_from_directory(
    directory=str(base_train),
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SZ,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

valid_ds = valid_gen.flow_from_directory(
    directory=str(base_valid),
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SZ,
    class_mode="categorical",
    shuffle=True,
    seed=123,
)

# ——— STEP 2 & 3: MODEL DESIGN + HYPERPARAMS ———
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

inp = Input(shape=INPUT_SHAPE, name="image")
x = Conv2D(16, (3, 3), activation="relu", name="conv_1")(inp)
x = MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)
x = Conv2D(32, (3, 3), activation="relu", name="conv_2")(x)
x = MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)
x = Flatten(name="flatten")(x)
x = Dense(32, activation="relu", name="fc_1")(x)
x = Dense(16, activation="relu", name="fc_2")(x)
out = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

model = Model(inputs=inp, outputs=out, name="baseline_cnn")
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ——— STEP 4: TRAIN / EVAL / SAVE ———
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    verbose=1,
)

model.save("model.keras")

# Plot training performance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Cross-Entropy Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
