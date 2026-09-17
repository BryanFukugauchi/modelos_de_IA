"""
train.py — Treino do classificador Derm Foundation sobre o HAM10000.

Diferente do EfficientNetV2/ConvNeXt, aqui NÃO se treina uma rede de
imagem fim-a-fim: o backbone (real ou substituto, ver model.py) fica
congelado e é usado só para gerar um vetor de características por
imagem. O que efetivamente é treinado é um classificador pequeno em
cima desses vetores.

Mudanças mais recentes:
  - Usa dataset_ham10000.py (módulo compartilhado) para treino/validação/
    teste — mesmo split de teste que EfficientNetV2/ConvNeXt, para
    comparação justa entre os três em comparar_modelos.py.
  - Os embeddings são calculados uma vez para o dataset inteiro e
    cacheados (EMBEDDINGS_CACHE); o split é aplicado depois, fatiando
    esse array pelos mesmos índices do dataframe.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset_ham10000 import load_ham10000_data, criar_splits

from model import (
    build_model,
    carregar_extrator_de_embeddings,
    carregar_extrator_substituto,
    imagem_para_embedding,
    imagem_para_embedding_substituto,
)

EMBEDDINGS_CACHE = "derm_embeddings_cache.npz"
USE_REAL_DERM_FOUNDATION = True  # Mude para False enquanto não tiver o acesso via Hugging Face configurado


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
    Calcula o embedding de cada imagem do dataframe completo, uma única
    vez, e guarda em disco. O resultado está na mesma ordem de df — para
    pegar só o pedaço de treino/validação/teste, fatie por df.index
    (ver train()).
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

    embeddings_completo = calcular_ou_carregar_embeddings(df)

    train_df, val_df, test_df = criar_splits(df)
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste (reservado): {len(test_df)}")

    # Fatia o array de embeddings pelos mesmos índices do split — garante
    # que treino/validação/teste sejam exatamente as mesmas imagens usadas
    # pelos outros modelos.
    X_train, y_train = embeddings_completo[train_df.index.values], train_df["label"].values
    X_val, y_val = embeddings_completo[val_df.index.values], val_df["label"].values

    class_weight = calcular_class_weight(y_train)
    print(f"Pesos de classe (compensando desbalanceamento): {class_weight}")

    model = build_model(num_classes=df["label"].nunique())
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
    print("Rode comparar_modelos.py depois de treinar os três para ver o desempenho no teste reservado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do classificador Derm Foundation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="derm_foundation_model.keras")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)
