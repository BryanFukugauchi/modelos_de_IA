"""
model.py — Arquitetura ConvNeXt (Tiny ou Small) para o HAM10000.

Mesmas mudanças aplicadas ao EfficientNetV2, pelo mesmo motivo
(overfitting: acurácia alta no treino, baixa em imagens novas):

  1. Aumento de dados (RandomFlip/Rotation/Zoom/Contrast).
  2. Backbone começa CONGELADO (trainable=False); train.py descongela
     as últimas camadas na fase 2 via descongelar_backbone().
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True, architecture="tiny"):
    """
    Constrói a arquitetura ConvNeXt (Tiny ou Small).

    Retorna (model, backbone): o backbone isolado é usado depois, em
    train.py, para descongelar as últimas camadas na fase de fine-tuning.
    """
    weights = 'imagenet' if pretrained else None

    if architecture.lower() == "small":
        base_model = tf.keras.applications.ConvNeXtSmall(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    else:  # Padrão: "tiny"
        base_model = tf.keras.applications.ConvNeXtTiny(
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

    model = models.Model(inputs, outputs, name=f"ConvNeXt_{architecture.capitalize()}")
    return model, base_model


def descongelar_backbone(backbone, camadas_para_descongelar=20):
    """Fase 2: destrava só as últimas N camadas do backbone."""
    backbone.trainable = True
    for camada in backbone.layers[:-camadas_para_descongelar]:
        camada.trainable = False
