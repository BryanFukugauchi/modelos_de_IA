import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True, architecture="tiny"):
    """
    Constrói a arquitetura ConvNeXt (Tiny ou Small) para 7 classes.
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

    base_model.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name=f"ConvNeXt_{architecture.capitalize()}")
    return model