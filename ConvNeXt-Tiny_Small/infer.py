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
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

def main(model_path="convnext_model.keras", image_path=None):
    print(f"Carregando modelo de: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if image_path:
        pred_class, confidence = predict(image_path, model)
        nome_classe = CLASSES_HAM10000.get(pred_class, "Desconhecido")
        print(f"Diagnóstico: {nome_classe} | Confiança: {confidence:.2%}")
    else:
        print("Modelo carregado com sucesso.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência com ConvNeXt")
    parser.add_argument("--model_path", type=str, default="convnext_model.keras")
    parser.add_argument("--image_path", type=str, default=None)
    args = parser.parse_args()

    main(model_path=args.model_path, image_path=args.image_path)