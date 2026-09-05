"""
train.py — Treino do classificador Derm Foundation sobre o HAM10000.

Diferente do EfficientNetV2/ConvNeXt, aqui NÃO se treina uma rede de
imagem fim-a-fim: o backbone (real ou substituto, ver model.py) fica
congelado e é usado só para gerar um vetor de características por
imagem. O que efetivamente é treinado é um classificador pequeno em
cima desses vetores — é o uso recomendado oficialmente para o Derm
Foundation.

Os embeddings são calculados uma vez e cacheados em disco
(EMBEDDINGS_CACHE), já que recalculá-los a cada execução seria caro.
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

from model import (
    build_model,
    carregar_extrator_de_embeddings,
    carregar_extrator_substituto,
    imagem_para_embedding,
    imagem_para_embedding_substituto,
)

EMBEDDINGS_CACHE = "derm_embeddings_cache.npz"
USE_REAL_DERM_FOUNDATION = True  # Mude para False enquanto não tiver o acesso via Hugging Face configurado


def load_ham10000_data():
    """Mesma lógica usada em EfficientNetV2/ConvNeXt — busca o CSV e mapeia as imagens."""
    csv_matches = glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)
    if not csv_matches:
        raise FileNotFoundError(
            "Não foi possível encontrar HAM10000_metadata.csv em /kaggle/input/."
        )

    metadata_path = csv_matches[0]
    base_dir = os.path.dirname(metadata_path)
    print(f"Dataset localizado com sucesso em: {base_dir}")

    df = pd.read_csv(metadata_path)

    all_image_paths = glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
    image_path_map = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}
    df["path"] = df["image_id"].map(image_path_map)

    # Descarta linhas cujo image_id não tem um .jpg correspondente no disco.
    linhas_sem_imagem = df["path"].isna().sum()
    if linhas_sem_imagem > 0:
        print(f"Aviso: {linhas_sem_imagem} imagens do CSV não foram encontradas no disco e serão ignoradas.")
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    df["label"] = pd.Categorical(df["dx"]).codes
    return df


def obter_funcao_de_embedding():
    """Escolhe o backbone real ou o substituto, deixando bem claro qual foi usado."""
    if USE_REAL_DERM_FOUNDATION:
        print(">> Usando o backbone REAL do Derm Foundation (Hugging Face).")
        infer_fn = carregar_extrator_de_embeddings()
        return lambda caminho: imagem_para_embedding(caminho, infer_fn)

    print(
        ">> AVISO: USE_REAL_DERM_FOUNDATION=False — usando backbone SUBSTITUTO "
        "(ConvNeXtTiny). Isto NÃO é o Derm Foundation real."
    )
    extrator = carregar_extrator_substituto()
    return lambda caminho: imagem_para_embedding_substituto(caminho, extrator)


def calcular_ou_carregar_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Calcula o embedding de cada imagem uma única vez e guarda em disco.
    Em execuções seguintes, se o cache já existir e tiver o mesmo número
    de imagens, ele é reaproveitado — evita recalcular tudo de novo.
    """
    if os.path.exists(EMBEDDINGS_CACHE):
        cache = np.load(EMBEDDINGS_CACHE)
        if len(cache["embeddings"]) == len(df):
            print(f"Reaproveitando cache de embeddings: {EMBEDDINGS_CACHE}")
            return cache["embeddings"]
        print("Cache de embeddings encontrado, mas com tamanho diferente do dataset atual — recalculando.")

    calcular_embedding = obter_funcao_de_embedding()

    embeddings = []
    for i, caminho in enumerate(df["path"], start=1):
        embeddings.append(calcular_embedding(caminho).numpy())
        if i % 200 == 0 or i == len(df):
            print(f"  {i}/{len(df)} embeddings calculados...")

    embeddings = np.stack(embeddings)
    np.savez(EMBEDDINGS_CACHE, embeddings=embeddings)
    print(f"Embeddings salvos em: {EMBEDDINGS_CACHE}")
    return embeddings


def calcular_class_weight(labels):
    """Compensa o desbalanceamento do HAM10000 ('nv' domina ~67% dos dados)."""
    classes = np.unique(labels)
    pesos = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), pesos.tolist()))


def train(epochs=20, batch_size=32, save_path="derm_foundation_model.keras"):
    print("Mapeando imagens do dataset HAM10000...")
    df = load_ham10000_data()
    print(f"Total de imagens encontradas: {len(df)}")

    embeddings = calcular_ou_carregar_embeddings(df)
    labels = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    class_weight = calcular_class_weight(y_train)
    print(f"Pesos de classe (compensando desbalanceamento): {class_weight}")

    model = build_model(num_classes=len(np.unique(labels)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True),
    ]

    print("Iniciando treinamento do classificador sobre os embeddings...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print(f"\nMelhor modelo (por val_accuracy) salvo automaticamente em: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do classificador Derm Foundation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="derm_foundation_model.keras")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)
