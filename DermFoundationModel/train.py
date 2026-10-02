"""
train.py — Treino do classificador Derm Foundation sobre o HAM10000.

Otimizações locais (Mac M1):
  - Carregamento automático de variáveis do arquivo .env (HF_TOKEN, DATASET_PATH).
  - Extração de embeddings paralelizada (Multithreading) na CPU.
  - Treinamento acelerado da cabeça classificadora em GPU (Apple Metal / CUDA) com fallback para CPU.
  - Reutilização automática do cache 'derm_embeddings_cache.npz'.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from dotenv import load_dotenv

# 1. Carrega variáveis de ambiente (.env)
load_dotenv()

# Autenticação e tokens do Hugging Face / OS
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# Garante a importação do dataset_ham10000 e model na raiz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_ham10000 import carregar_dataset, criar_splits
from model import (
    build_model,
    carregar_extrator_de_embeddings,
    carregar_extrator_substituto,
    imagem_para_embedding_substituto,
)

EMBEDDINGS_CACHE = "derm_embeddings_cache.npz"
USE_REAL_DERM_FOUNDATION = True  # Mude para False se quiser testar com o modelo substituto


def preparar_exemplo_png(caminho_imagem: str) -> bytes:
    """Helper para converter imagens no formato do Derm Foundation usando multithreading."""
    DERM_FOUNDATION_INPUT_SIZE = (448, 448)
    img = Image.open(caminho_imagem).convert("RGB").resize(DERM_FOUNDATION_INPUT_SIZE)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    exemplo = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[buffer.getvalue()])
                )
            }
        )
    ).SerializeToString()
    return exemplo


def calcular_ou_carregar_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Calcula os embeddings de cada imagem em lotes (paralelizado) e salva no disco.
    """
    if os.path.exists(EMBEDDINGS_CACHE):
        cache = np.load(EMBEDDINGS_CACHE)
        if len(cache["embeddings"]) == len(df):
            print(f">> Reaproveitando cache de embeddings local: {EMBEDDINGS_CACHE}")
            return cache["embeddings"]
        print(">> Cache com tamanho diferente do dataset atual — recalculando...")

    caminhos = df["path"].tolist()
    total = len(caminhos)
    embeddings = []

    if USE_REAL_DERM_FOUNDATION:
        print(">> Usando o backbone REAL do Derm Foundation.")
        print(">> Iniciando extração rápida em lote (Multithreading)...")
       
        infer_fn = carregar_extrator_de_embeddings()
        batch_size = 64

        for i in range(0, total, batch_size):
            lote_caminhos = caminhos[i : i + batch_size]
           
            # Paraleliza a leitura e conversão das imagens nas CPUs
            with ThreadPoolExecutor() as executor:
                exemplos_lote = list(executor.map(preparar_exemplo_png, lote_caminhos))
           
            # Executa a inferência de embedding na CPU
            with tf.device("/CPU:0"):
                saida = infer_fn(inputs=tf.constant(exemplos_lote))
                embeddings_lote = saida["embedding"].numpy()
           
            embeddings.append(embeddings_lote)
           
            progresso = min(i + batch_size, total)
            print(f"  {progresso}/{total} embeddings calculados...")
       
        embeddings = np.vstack(embeddings)

    else:
        print(">> AVISO: Usando backbone SUBSTITUTO (ConvNeXtTiny).")
        extrator = carregar_extrator_substituto()
        for i, caminho in enumerate(caminhos, start=1):
            emb = imagem_para_embedding_substituto(caminho, extrator).numpy()
            embeddings.append(emb)
            if i % 200 == 0 or i == total:
                print(f"  {i}/{total} embeddings calculados...")
       
        embeddings = np.stack(embeddings)

    np.savez(EMBEDDINGS_CACHE, embeddings=embeddings)
    print(f">> Embeddings salvos com sucesso em: {EMBEDDINGS_CACHE}")
    return embeddings


def calcular_class_weight(labels):
    """Compensa o desbalanceamento das classes do HAM10000."""
    classes = np.unique(labels)
    pesos = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), pesos.tolist()))


def train(epochs=50, batch_size=32, save_path="derm_foundation_model.keras"):
    print("Mapeando imagens do dataset HAM10000...")
    # Usa a função carregar_dataset() configurada via .env
    df = carregar_dataset()
    print(f"Total de imagens carregadas: {len(df)}")

    # MODO_DEBUG: Mude para True apenas se quiser testar a pipeline completa com 100 fotos
    MODO_DEBUG = False
    if MODO_DEBUG:
        print("\n>> ATENÇÃO: MODO_DEBUG ATIVADO! Usando apenas 100 imagens.\n")
        df = df.head(100)

    embeddings_completo = calcular_ou_carregar_embeddings(df)

    train_df, val_df, test_df = criar_splits(df)
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste (reservado): {len(test_df)}")

    # Fatia os embeddings pelos mesmos índices dos splits do dataframe
    X_train, y_train = embeddings_completo[train_df.index.values], train_df["label"].values
    X_val, y_val = embeddings_completo[val_df.index.values], val_df["label"].values

    class_weight = calcular_class_weight(y_train)
    print(f"Pesos de classe calculados: {class_weight}")

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

    # Detecta automaticamente se há GPU disponível (Apple Metal MPS ou CUDA)
    gpus = tf.config.list_physical_devices('GPU')
    device_treino = "/GPU:0" if gpus else "/CPU:0"
    print(f"Iniciando treinamento do classificador no dispositivo: {device_treino}...")

    with tf.device(device_treino):
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
        )

    print(f"\nMelhor modelo salvo com sucesso em: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do classificador Derm Foundation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="derm_foundation_model.keras")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)