import tensorflow as tf
import gradio as gr
import numpy as np

# Charger le modèle
model = tf.keras.models.load_model('FossilesClassification.h5')

# Définir la fonction de prédiction avec des messages de débogage
def predict(image):
    try:
        # Redimensionner l'image
        image = tf.image.resize(image, (256, 256))
        print(f"Image redimensionnée : {image.shape}")

        # Normaliser l'image
        image = image / 255.0
        print(f"Image normalisée : {image.shape}")

        # Ajouter une dimension batch
        image = tf.expand_dims(image, axis=0)
        print(f"Image avec dimension batch : {image.shape}")

        # Faire la prédiction
        prediction = model.predict(image)
        print(f"Prédiction brute : {prediction}")

        # Vérifier le type et la forme de la prédiction
        if isinstance(prediction, np.ndarray):
            print(f"Prédiction est un tableau numpy : {prediction.shape}")

        # Retourner l'index de la classe prédite
        predicted_class = int(np.argmax(prediction))
        print(f"Classe prédite : {predicted_class}")
        return predicted_class
    except Exception as e:
        print(f"Erreur dans la fonction de prédiction : {str(e)}")
        return "Erreur dans la prédiction"

# Créer l'interface Gradio
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="numpy"),
                         outputs=gr.Label(num_top_classes=1),
                         title="Fossile Detection")

# Lancer l'interface
if __name__ == "__main__":
    interface.launch(share=True)

