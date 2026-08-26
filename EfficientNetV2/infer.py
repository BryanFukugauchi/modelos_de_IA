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

def predict(image_path, model):
    # Carrega e pré-processa a imagem real do disco
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

def main(model_path="ham10000_efficientnetv2.keras", image_path=None):
    model = tf.keras.models.load_model(model_path, compile=False)
    if image_path:
        pred_class, confidence = predict(image_path, model)
        print(f"Diagnóstico: {CLASSES_HAM10000.get(pred_class, 'Desconhecido')} | Confiança: {confidence:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ham10000_efficientnetv2.keras")
    parser.add_argument("--image_path", type=str, required=True, help="Caminho para uma imagem real (.jpg / .png)")
    args = parser.parse_args()

    main(model_path=args.model_path, image_path=args.image_path)