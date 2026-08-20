import argparse
import tensorflow as tf
import numpy as np

CLASSES_HAM10000 = {
    0: "akiec - Queratose Actínica",
    1: "bcc - Carcinoma Basocelular",
    2: "bkl - Lesão Benigna do Tipo Queratose",
    3: "df - Dermatofibroma",
    4: "mel - Melanoma",
    5: "nv - Nevo Melanocítico (Pinta comum)",
    6: "vasc - Lesão Vascular"
}

def predict(image_array, model):
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

def main(model_path="efficientnet_v2_model.keras"):
    print(f"Carregando modelo de: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Simula uma imagem de entrada [224, 224, 3]
    dummy_input = np.random.randn(224, 224, 3).astype(np.float32)

    pred_class, confidence = predict(dummy_input, model)
    print(f"Classe prevista: {pred_class} | Confiança: {confidence:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência com EfficientNetV2")
    parser.add_argument("--model_path", type=str, default="efficientnet_v2_model.keras", help="Caminho do arquivo .keras")
    args = parser.parse_args()

    main(model_path=args.model_path)