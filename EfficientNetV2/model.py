import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True):
    """
    Constrói a arquitetura do modelo usando EfficientNetV2-S.
    """
    weights = 'imagenet' if pretrained else None

    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    base_model.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model