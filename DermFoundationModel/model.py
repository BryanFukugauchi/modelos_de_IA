import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True):
    """
    Carrega o Derm Foundation diretamente do diretório do Kaggle ou via Kaggle Models URL.
    """
    import tensorflow_hub as hub

    # Caminho local quando o modelo é adicionado via botão '+ Add Model' no Kaggle
    local_path = "/kaggle/input/derm-foundation/tensorflow2/default/1"
    # URL oficial atualizada do Kaggle Models
    remote_url = "https://www.kaggle.com/models/google/derm-foundation/TensorFlow2/default/1"

    inputs = tf.keras.Input(shape=input_shape)
    
    # Pre-processamento exigido pelo Derm Foundation (pixels [0, 1])
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    if os.path.exists(local_path):
        print(f"Carregando Derm Foundation localmente de: {local_path}")
        model_source = local_path
    else:
        print(f"Carregando Derm Foundation remoto de: {remote_url}")
        model_source = remote_url

    base_layer = hub.KerasLayer(model_source, trainable=pretrained, name="derm_foundation_backbone")
    features = base_layer(x)

    x = layers.Dense(256, activation='relu')(features)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="DermFoundation_HAM10000")
    return model