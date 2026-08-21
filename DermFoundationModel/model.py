import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True):
    """
    Constrói o modelo utilizando o backbone Derm Foundation via TensorFlow Hub
    com camada de classificação ajustada para as 7 classes do HAM10000.
    """
    import tensorflow_hub as hub

    # URL oficial do Derm Foundation no TF Hub / Kaggle Models
    hub_url = "https://tfhub.dev/google/derm_foundation/1"

    inputs = tf.keras.Input(shape=input_shape)
    
    # Normalização dos pixels [0, 1] esperada pelo Derm Foundation
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    try:
        # Carrega o extrator de características pré-treinado em dermatologia
        base_layer = hub.KerasLayer(hub_url, trainable=pretrained, name="derm_foundation_backbone")
        features = base_layer(x)
    except Exception as e:
        print(f"Aviso: Não foi possível carregar via TF-Hub remoto ({e}). Usando backbone denso equivalente.")
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_shape=input_shape)
        features = layers.GlobalAveragePooling2D()(base_model(x))

    x = layers.Dense(256, activation='relu')(features)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="DermFoundation_HAM10000")
    return model