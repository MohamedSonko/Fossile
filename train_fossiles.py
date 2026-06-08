"""
Entraînement amélioré pour la classification des fossiles (11 classes).

Stratégie anti-underfitting :
  1. Transfer learning (MobileNetV2 pré-entraîné ImageNet) -> features de qualité
     malgré le très petit dataset (~48 images / classe).
  2. Data augmentation -> multiplie virtuellement les données.
  3. BatchNormalization + tête dense correctement dimensionnée.
  4. Entraînement en 2 phases : (a) tête seule, (b) fine-tuning du backbone.
  5. Class weights -> compense le léger déséquilibre des classes.
  6. Callbacks (EarlyStopping, ReduceLROnPlateau) -> on entraîne assez longtemps
     sans gaspiller, et on garde le meilleur modèle.

Usage :
    python train_fossiles.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = "Fossiles"
IMG_SIZE = (224, 224)          # taille native de MobileNetV2
BATCH_SIZE = 16                # petit dataset -> petit batch
SEED = 42
EPOCHS_HEAD = 25               # phase 1 : tête seule
EPOCHS_FINETUNE = 25           # phase 2 : fine-tuning
AUTOTUNE = tf.data.AUTOTUNE

# ----------------------------------------------------------------------------
# 1. Chargement des données (split train/validation 80/20)
# ----------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n{num_classes} classes : {class_names}\n")

# Sauvegarde de l'ordre des classes pour l'inférence (app.py)
with open("class_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

# ----------------------------------------------------------------------------
# 2. Class weights (compense bellerophon ~31 vs autres ~50)
# ----------------------------------------------------------------------------
y_train = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print("Class weights :", class_weight)

# Performance pipeline
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ----------------------------------------------------------------------------
# 3. Augmentation de données
# ----------------------------------------------------------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# ----------------------------------------------------------------------------
# 4. Modèle (transfer learning)
# ----------------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # gelé en phase 1

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # mise à l'échelle attendue
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)                     # dropout modéré (pas 0.5)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

# ----------------------------------------------------------------------------
# 5. Phase 1 : entraînement de la tête
# ----------------------------------------------------------------------------
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

cb = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=4, min_lr=1e-6),
]

print("\n=== Phase 1 : entraînement de la tête ===")
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                  class_weight=class_weight, callbacks=cb)

# ----------------------------------------------------------------------------
# 6. Phase 2 : fine-tuning du backbone (couches hautes)
# ----------------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-40]:   # on ne dégèle que les ~40 dernières couches
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),     # LR très faible pour le fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n=== Phase 2 : fine-tuning ===")
hist2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
                  class_weight=class_weight, callbacks=cb)

# ----------------------------------------------------------------------------
# 7. Sauvegarde + courbes
# ----------------------------------------------------------------------------
model.save("model/FossilesClassification.h5")
print("\nModèle sauvegardé -> model/FossilesClassification.h5")

acc = hist1.history["accuracy"] + hist2.history["accuracy"]
val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
loss = hist1.history["loss"] + hist2.history["loss"]
val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="train"); plt.plot(val_acc, label="val")
plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label="train"); plt.plot(val_loss, label="val")
plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
print("Courbes -> training_curves.png")
