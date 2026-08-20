"""
Transfer learning básico com TensorFlow/Keras: adapta um backbone
pré-treinado (EfficientNetV2 ou ConvNeXt) para as SUAS classes.

Estrutura de pastas esperada (ajuste DATASET_DIR se a sua for diferente):

    dataset/
        train/
            classe_a/
                img1.jpg
                img2.jpg
            classe_b/
                ...
        val/
            classe_a/
                ...
            classe_b/
                ...

tf.keras.utils.image_dataset_from_directory usa o nome de cada subpasta
como nome (e rótulo) da classe automaticamente.

Duas estratégias de transfer learning, controladas por FAZER_FINE_TUNING:

    1) Extração de características (FAZER_FINE_TUNING = False)
       Congela TODO o backbone pré-treinado e treina só a "cabeça" nova.
       Rápido, funciona com pouco dado, baixo risco de overfitting.
       Bom baseline inicial.

    2) Extração + Fine-tuning (FAZER_FINE_TUNING = True)
       Depois de treinar a cabeça, descongela as últimas camadas do
       backbone e continua o treino com uma taxa de aprendizado bem
       menor. Tende a dar mais acurácia, mas exige mais dados e cuidado
       (LR alta aqui destrói os pesos pré-treinados).

Requisitos:
    pip install tensorflow pillow numpy

Uso:
    python transfer_learning_basico.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# Configuração — ajuste aqui, sem precisar tocar no resto do script.
# ----------------------------------------------------------------------
DATASET_DIR = Path("dataset")
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 32

# Escolha o backbone trocando esta string: "efficientnetv2" ou "convnext"
BACKBONE = "efficientnetv2"

EPOCHS_CABECA = 10              # Fase 1: só a cabeça nova
EPOCHS_FINE_TUNING = 5          # Fase 2: cabeça + últimas camadas do backbone
FAZER_FINE_TUNING = True        # Alterne aqui entre Opção 1 e Opção 2
CAMADAS_PARA_DESCONGELAR = 20   # Quantas camadas finais liberar na fase 2
LR_CABECA = 1e-3
LR_FINE_TUNING = 1e-5           # Bem menor — protege os pesos pré-treinados


def carregar_datasets():
    """
    Lê as imagens direto das pastas (train/ e val/) e devolve os
    datasets já no formato que o Keras consome durante o treino.

    Observação: as imagens saem em pixels [0, 255] (float32), que é
    justamente o que EfficientNetV2 e ConvNeXt esperam receber, já que
    os dois têm a normalização embutida no próprio modelo — por isso
    não fazemos nenhum rescale manual aqui.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    return train_ds, val_ds


def criar_backbone() -> tf.keras.Model:
    """
    Cria o backbone pré-treinado SEM a camada de classificação original
    (include_top=False) — é essa camada que vamos substituir pelas
    nossas próprias classes. pooling="avg" já resume o mapa de
    características final em um único vetor por imagem.
    """
    if BACKBONE == "efficientnetv2":
        return tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights="imagenet", pooling="avg"
        )
    if BACKBONE == "convnext":
        return tf.keras.applications.ConvNeXtTiny(
            include_top=False, weights="imagenet", pooling="avg"
        )
    raise ValueError(f"Backbone desconhecido: {BACKBONE}")


def criar_modelo(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Monta o modelo completo: aumento de dados -> backbone -> cabeça nova.
    Retorna também uma referência isolada ao backbone, para poder
    congelar/descongelar suas camadas na fase de fine-tuning.
    """
    aumento_de_dados = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="aumento_de_dados",
    )

    backbone = criar_backbone()
    backbone.trainable = False  # Fase 1: backbone inteiro congelado

    entradas = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = aumento_de_dados(entradas)
    x = backbone(x, training=False)
    x = layers.Dropout(0.2)(x)
    saidas = layers.Dense(num_classes, activation="softmax")(x)

    modelo = tf.keras.Model(entradas, saidas)
    return modelo, backbone


def compilar_e_treinar_cabeca(modelo, train_ds, val_ds):
    """Fase 1: treina só a cabeça nova, com o backbone congelado."""
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR_CABECA),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== Fase 1: treinando a cabeça (backbone congelado) ===")
    return modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_CABECA)


def fazer_fine_tuning(modelo, backbone, train_ds, val_ds):
    """
    Fase 2 (opcional): descongela as últimas camadas do backbone e
    continua o treino com uma taxa de aprendizado bem menor, ajustando
    essas camadas ao seu domínio sem destruir o que o modelo já sabe.
    """
    backbone.trainable = True
    for camada in backbone.layers[:-CAMADAS_PARA_DESCONGELAR]:
        camada.trainable = False

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE_TUNING),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== Fase 2: fine-tuning das últimas camadas do backbone ===")
    return modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNING)


def main() -> None:
    train_ds, val_ds = carregar_datasets()
    nomes_das_classes = train_ds.class_names
    print(f"Classes encontradas: {nomes_das_classes}")

    modelo, backbone = criar_modelo(num_classes=len(nomes_das_classes))
    compilar_e_treinar_cabeca(modelo, train_ds, val_ds)

    if FAZER_FINE_TUNING:
        fazer_fine_tuning(modelo, backbone, train_ds, val_ds)

    modelo.save("modelo_treinado.keras")
    print("\nModelo salvo em 'modelo_treinado.keras'.")


if __name__ == "__main__":
    main()
