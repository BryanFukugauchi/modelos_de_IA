"""
train.py — Treino do ConvNeXt (Tiny/Small) no HAM10000, em duas fases.

Mesmas mudanças aplicadas ao EfficientNetV2, pelo mesmo motivo
(overfitting: acurácia alta no treino, baixa em imagens novas):

  1. Split ESTRATIFICADO.
  2. class_weight para compensar o desbalanceamento do HAM10000.
  3. Treino em DUAS FASES: cabeça (backbone congelado) e depois
     fine-tuning das últimas camadas com learning rate bem menor.
  4. EarlyStopping + ModelCheckpoint(save_best_only=True).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import build_model, descongelar_backbone


def load_ham10000_data():
    """Busca automaticamente o arquivo HAM10000_metadata.csv em qualquer pasta do Kaggle."""
    csv_matches = glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)

    if not csv_matches:
        raise FileNotFoundError(
            "Não foi possível encontrar HAM10000_metadata.csv em /kaggle/input/. "
            "Certifique-se de adicionar o dataset HAM10000 no menu lateral do Kaggle."
        )

    metadata_path = csv_matches[0]
    base_dir = os.path.dirname(metadata_path)
    print(f"Dataset localizado com sucesso em: {base_dir}")

    df = pd.read_csv(metadata_path)

    all_image_paths = glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
    image_path_map = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths
    }
    df["path"] = df["image_id"].map(image_path_map)

    linhas_sem_imagem = df["path"].isna().sum()
    if linhas_sem_imagem > 0:
        print(f"Aviso: {linhas_sem_imagem} imagens do CSV não foram encontradas no disco e serão ignoradas.")
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    df["label"] = pd.Categorical(df["dx"]).codes
    return df


def create_tf_dataset(df, batch_size=32, img_size=(224, 224), embaralhar=True):
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, img_size)
        return image, label

    filenames = df["path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if embaralhar:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def calcular_class_weight(labels):
    """Compensa o desbalanceamento do HAM10000 ('nv' domina ~67% dos dados)."""
    classes = np.unique(labels)
    pesos = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), pesos.tolist()))


def train(
    epochs_cabeca=10,
    epochs_fine_tuning=10,
    batch_size=32,
    architecture="tiny",
    camadas_para_descongelar=20,
    save_path="convnext_model.keras",
):
    print("Mapeando imagens do dataset HAM10000...")
    df = load_ham10000_data()
    print(f"Total de imagens encontradas: {len(df)}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Treino: {len(train_df)} imagens | Validação: {len(val_df)} imagens")

    train_ds = create_tf_dataset(train_df, batch_size=batch_size, embaralhar=True)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, embaralhar=False)

    class_weight = calcular_class_weight(train_df["label"].values)
    print(f"Pesos de classe (compensando desbalanceamento): {class_weight}")

    print(f"Construindo modelo ConvNeXt ({architecture.upper()})...")
    model, backbone = build_model(
        input_shape=(224, 224, 3), num_classes=7, pretrained=True, architecture=architecture
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True),
    ]

    # --- Fase 1: treina só a cabeça, com o backbone congelado ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\n=== Fase 1: treinando a cabeça (backbone congelado) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_cabeca,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # --- Fase 2: destrava as últimas camadas do backbone e ajusta fino ---
    descongelar_backbone(backbone, camadas_para_descongelar)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\n=== Fase 2: fine-tuning das últimas camadas do backbone ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_fine_tuning,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print(f"\nMelhor modelo (por val_accuracy) salvo automaticamente em: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do ConvNeXt para HAM10000")
    parser.add_argument("--epochs_cabeca", type=int, default=10)
    parser.add_argument("--epochs_fine_tuning", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--architecture", type=str, default="tiny", choices=["tiny", "small"])
    parser.add_argument("--camadas_para_descongelar", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="convnext_model.keras")
    args = parser.parse_args()

    train(
        epochs_cabeca=args.epochs_cabeca,
        epochs_fine_tuning=args.epochs_fine_tuning,
        batch_size=args.batch_size,
        architecture=args.architecture,
        camadas_para_descongelar=args.camadas_para_descongelar,
        save_path=args.save_path,
    )
