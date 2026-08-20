import argparse
import tensorflow as tf
import numpy as np
from model import build_model

def train_pipeline(epochs=1, batch_size=8, save_path="efficientnet_v2_model.keras"):
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs disponíveis para uso: {len(gpus)}")

    num_classes = 2
    model = build_model(input_shape=(224, 224, 3), num_classes=num_classes, pretrained=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Dados sintéticos de exemplo (32 imagens [224, 224, 3])
    dummy_x = np.random.randn(32, 224, 224, 3).astype(np.float32)
    dummy_y = np.random.randint(0, num_classes, size=(32,))

    print("Iniciando o treinamento...")
    model.fit(dummy_x, dummy_y, batch_size=batch_size, epochs=epochs)

    # Salva o modelo treinado
    model.save(save_path)
    print(f"Modelo salvo com sucesso em: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do EfficientNetV2")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--save_path", type=str, default="efficientnet_v2_model.keras", help="Caminho para salvar")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)