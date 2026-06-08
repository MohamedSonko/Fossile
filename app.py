import os
import numpy as np
import tensorflow as tf
import gradio as gr

# ---------------------------------------------------------------------------
# 0. Correctif du bug gradio_client sous Python 3.9 / Gradio 4.44.1
#    ("TypeError: argument of type 'bool' is not iterable")
#    On rend la génération du schéma d'API tolérante aux schémas booléens.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
_orig_j2p = _gc_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


def _safe_j2p(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_j2p(schema, defs)


_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_j2p

# ---------------------------------------------------------------------------
# 1. Chargement du modèle (racine ou dossier model/)
# ---------------------------------------------------------------------------
MODEL_PATH = "model/FossilesClassification.h5"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "FossilesClassification.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# Taille d'entrée déduite AUTOMATIQUEMENT du modèle -> aucun décalage possible
# (224x224 pour le modèle transfer learning, 256x256 pour l'ancien modèle).
_, H, W, _ = model.input_shape
IMG_SIZE = (H, W)
print(f"Modèle chargé : {MODEL_PATH} | entrée attendue : {IMG_SIZE}")

# ---------------------------------------------------------------------------
# 2. Noms de classes (ordre figé à l'entraînement par train_fossiles.py)
# ---------------------------------------------------------------------------
if os.path.exists("class_names.txt"):
    with open("class_names.txt", encoding="utf-8") as f:
        CLASS_NAMES = [l.strip() for l in f if l.strip()]
else:
    # repli : ordre alphabétique des dossiers (= ordre de image_dataset_from_directory)
    CLASS_NAMES = sorted(d for d in os.listdir("Fossiles")
                         if os.path.isdir(os.path.join("Fossiles", d)))

# Vérification de cohérence classes <-> sorties du modèle
n_out = model.output_shape[-1]
assert len(CLASS_NAMES) == n_out, (
    f"{len(CLASS_NAMES)} noms de classes mais le modèle a {n_out} sorties. "
    "Vérifiez class_names.txt."
)
print(f"{len(CLASS_NAMES)} classes : {CLASS_NAMES}")


# ---------------------------------------------------------------------------
# 3. Fonction de prédiction
# ---------------------------------------------------------------------------
def predict(image):
    """Retourne {nom_fossile: probabilité} -> gr.Label affiche le nom + score."""
    if image is None:
        return {}
    # La normalisation (preprocess_input) est intégrée au graphe du modèle :
    # on passe donc l'image brute 0-255, simplement redimensionnée.
    img = tf.image.resize(image, IMG_SIZE)
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}


# ---------------------------------------------------------------------------
# 4. Interface Gradio
# ---------------------------------------------------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Image du fossile"),
    outputs=gr.Label(num_top_classes=3, label="Classe prédite"),
    title="🦕 Détection de fossiles",
    description=(
        "Chargez une image de fossile. Le modèle affiche les 3 classes les plus "
        "probables avec leur nom et leur niveau de confiance."
    ),
)

if __name__ == "__main__":
    interface.launch(share=True)
