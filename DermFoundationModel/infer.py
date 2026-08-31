"""
infer.py — Inferência com o classificador Derm Foundation.

Repete o mesmo caminho do treino: calcula o embedding da imagem (via o
backbone real ou substituto — mesma flag USE_REAL_DERM_FOUNDATION de
train.py, importada daqui para nunca ficar dessincronizada) e passa
pelo classificador leve treinado.
"""

import argparse

import numpy as np
import tensorflow as tf

from model import (
    carregar_extrator_de_embeddings,
    carregar_extrator_substituto,
    imagem_para_embedding,
    imagem_para_embedding_substituto,
)
from train import USE_REAL_DERM_FOUNDATION

CLASSES_HAM10000 = {
    0: "akiec - Queratose Actínica",
    1: "bcc - Carcinoma Basocelular",
    2: "bkl - Lesão Benigna do Tipo Queratose",
    3: "df - Dermatofibroma",
    4: "mel - Melanoma",
    5: "nv - Nevo Melanocítico (Pinta comum)",
    6: "vasc - Lesão Vascular",
}


def obter_funcao_de_embedding():
    if USE_REAL_DERM_FOUNDATION:
        infer_fn = carregar_extrator_de_embeddings()
        return lambda caminho: imagem_para_embedding(caminho, infer_fn)

    print(">> AVISO: usando backbone SUBSTITUTO (ConvNeXtTiny), não o Derm Foundation real.")
    extrator = carregar_extrator_substituto()
    return lambda caminho: imagem_para_embedding_substituto(caminho, extrator)


def predict(image_path, model):
    calcular_embedding = obter_funcao_de_embedding()
    embedding = calcular_embedding(image_path).numpy()
    embedding = np.expand_dims(embedding, axis=0)

    predictions = model.predict(embedding, verbose=0)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(predictions[0][predicted_class])

    return predicted_class, confidence


def main(model_path="derm_foundation_model.keras", image_path=None):
    print(f"Carregando modelo de: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    if image_path:
        pred_class, confidence = predict(image_path, model)
        nome_classe = CLASSES_HAM10000.get(pred_class, "Desconhecido")
        print(f"Diagnóstico: {nome_classe} | Confiança: {confidence:.2%}")
    else:
        print("Modelo carregado com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência com Derm Foundation")
    parser.add_argument("--model_path", type=str, default="derm_foundation_model.keras")
    parser.add_argument("--image_path", type=str, default=None)
    args = parser.parse_args()

    main(model_path=args.model_path, image_path=args.image_path)
