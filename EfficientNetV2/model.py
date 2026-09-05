"""
model.py — Arquitetura do EfficientNetV2-S para o HAM10000.

Mudanças em relação à versão anterior (motivadas pelo overfitting
observado: acurácia alta no treino, baixa em imagens novas):

  1. Aumento de dados (RandomFlip/Rotation/Zoom/Contrast) — reduz a
     capacidade do modelo de simplesmente decorar as imagens de treino.
  2. Backbone começa CONGELADO (trainable=False). O treino em duas fases
     (cabeça primeiro, depois destravar as últimas camadas) acontece em
     train.py, chamando descongelar_backbone() na fase 2.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True):
    """
    Constrói a arquitetura do modelo usando EfficientNetV2-S.

    Retorna (model, backbone): o backbone isolado é usado depois, em
    train.py, para descongelar as últimas camadas na fase de fine-tuning.
    """
    weights = 'imagenet' if pretrained else None

    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    base_model.trainable = False  # Fase 1: começa congelado

    aumento_de_dados = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="aumento_de_dados")

    inputs = tf.keras.Input(shape=input_shape)
    x = aumento_de_dados(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def descongelar_backbone(backbone, camadas_para_descongelar=20):
    """
    Fase 2: destrava só as últimas N camadas do backbone, mantendo o
    restante congelado — permite ajuste fino sem destruir os pesos
    pré-treinados das camadas mais profundas (as que capturam
    características mais genéricas, como bordas e texturas).
    """
    backbone.trainable = True
    for camada in backbone.layers[:-camadas_para_descongelar]:
        camada.trainable = False
