"""
train.py — Treino do EfficientNetV2-S no HAM10000, em duas fases.

Mudanças mais recentes:
  - Agora usa dataset_ham10000.py (módulo compartilhado) em vez de uma
    cópia própria de load_ham10000_data() — corrige o bug uma vez em
    vez de três, e garante que treino/validação/teste sejam os MESMOS
    conjuntos usados pelos outros modelos, para comparação justa.
  - Split de 3 vias (treino/validação/teste): o teste nunca é visto
    durante o treino, nem usado para EarlyStopping/ModelCheckpoint —
    fica reservado para comparar_modelos.py no final.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset_ham10000 import load_ham10000_data, criar_splits

from model import build_model, descongelar_backbone


def create_tf_dataset(df, batch_size=32, img_size=(224, 224), embaralhar=True):
    """Carrega as imagens do disco e redimensiona para 224x224 em tempo de execução."""
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, img_size)  # bilinear (padrão) — precisa bater com infer.py
        return image, label

    filenames = df["path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if embaralhar:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def calcular_class_weight(labels):
    """HAM10000 é extremamente desbalanceado ('nv' domina ~67% dos dados)."""
    classes = np.unique(labels)
    pesos = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), pesos.tolist()))


def train(
    epochs_cabeca=10,
    epochs_fine_tuning=10,
    batch_size=32,
    camadas_para_descongelar=20,
    save_path="ham10000_efficientnetv2.keras",
):
    print("Mapeando imagens do dataset HAM10000...")
    df = load_ham10000_data()
    print(f"Total de imagens encontradas: {len(df)}")

    train_df, val_df, test_df = criar_splits(df)
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste (reservado): {len(test_df)}")

    train_ds = create_tf_dataset(train_df, batch_size=batch_size, embaralhar=True)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, embaralhar=False)

    class_weight = calcular_class_weight(train_df["label"].values)
    print(f"Pesos de classe (compensando desbalanceamento): {class_weight}")

    model, backbone = build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\n=== Fase 1: treinando a cabeça (backbone congelado) ===")
    historico_fase1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_cabeca,
        class_weight=class_weight, callbacks=callbacks,
    )
    melhor_val_acc_fase1 = max(historico_fase1.history["val_accuracy"])

    descongelar_backbone(backbone, camadas_para_descongelar)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\n=== Fase 2: fine-tuning das últimas camadas do backbone ===")
    callbacks_fase2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy", save_best_only=True,
            initial_value_threshold=melhor_val_acc_fase1,
        ),
    ]
    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_fine_tuning,
        class_weight=class_weight, callbacks=callbacks_fase2,
    )

    print(f"\nMelhor modelo (por val_accuracy) salvo automaticamente em: {save_path}")
    print("Rode comparar_modelos.py depois de treinar os três para ver o desempenho no teste reservado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_cabeca", type=int, default=10)
    parser.add_argument("--epochs_fine_tuning", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--camadas_para_descongelar", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="ham10000_efficientnetv2.keras")
    args = parser.parse_args()

    train(
        epochs_cabeca=args.epochs_cabeca,
        epochs_fine_tuning=args.epochs_fine_tuning,
        batch_size=args.batch_size,
        camadas_para_descongelar=args.camadas_para_descongelar,
        save_path=args.save_path,
    )
