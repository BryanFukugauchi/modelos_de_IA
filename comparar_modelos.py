"""
comparar_modelos.py — Avalia os três modelos treinados (EfficientNetV2,
ConvNeXt, Derm Foundation) no MESMO conjunto de teste, nunca visto
durante nenhum treino, e gera um gráfico comparando o desempenho.

Por que isso importa: até agora, cada modelo era testado manualmente,
com imagens diferentes — não dá para dizer com confiança qual é o
melhor assim. Este script usa o split de teste fixo de
dataset_ham10000.py (mesmas imagens para os três) e reporta duas
métricas por modelo:

  - accuracy: percentual de acertos geral.
  - f1_macro: média do F1 por classe, sem peso pela frequência. Como o
    HAM10000 é muito desbalanceado, essa métrica é mais justa — um
    modelo que "chuta" sempre a classe majoritária tem accuracy alta
    mas f1_macro baixo, porque erra muito nas classes raras (como
    melanoma, que é a mais importante clinicamente).

Requisitos: pip install matplotlib scikit-learn (os demais já são
usados pelos três modelos).

Uso:
    python comparar_modelos.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from dataset_ham10000 import load_ham10000_data, criar_splits

PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))

CAMINHOS_MODELOS = {
    "EfficientNetV2": os.path.join(PASTA_RAIZ, "EfficientNetV2", "ham10000_efficientnetv2.keras"),
    "ConvNeXt": os.path.join(PASTA_RAIZ, "ConvNeXt-Tiny_Small", "convnext_model.keras"),
    "Derm Foundation": os.path.join(PASTA_RAIZ, "DermFoundationModel", "derm_foundation_model.keras"),
}


def construir_dataset_teste(test_df, batch_size=32, img_size=(224, 224)):
    """
    Mesmo pipeline de pré-processamento usado no treino (tf.image.resize,
    bilinear) — garante que a avaliação não sofre do mesmo tipo de
    inconsistência treino/inferência já corrigida no infer.py de cada
    modelo.
    """
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, img_size)
        return image, label

    filenames = test_df["path"].values
    labels = test_df["label"].values
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, labels


def avaliar_modelo_cnn(caminho_modelo, test_df):
    """Avalia o EfficientNetV2 ou o ConvNeXt — ambos são classificadores de imagem diretos."""
    if not os.path.exists(caminho_modelo):
        return None

    model = tf.keras.models.load_model(caminho_modelo)
    test_ds, y_true = construir_dataset_teste(test_df)
    predictions = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    return np.array(y_true), y_pred


def avaliar_derm_foundation(caminho_modelo, test_df):
    """
    Avalia o Derm Foundation — precisa primeiro converter cada imagem em
    embedding (via o mesmo backbone real/substituto usado no treino),
    já que o classificador salvo espera embeddings, não pixels.
    """
    if not os.path.exists(caminho_modelo):
        return None

    pasta_derm = os.path.join(PASTA_RAIZ, "DermFoundationModel")
    if pasta_derm not in sys.path:
        sys.path.insert(0, pasta_derm)

    from model import (
        carregar_extrator_de_embeddings,
        carregar_extrator_substituto,
        imagem_para_embedding,
        imagem_para_embedding_substituto,
    )
    from train import USE_REAL_DERM_FOUNDATION

    if USE_REAL_DERM_FOUNDATION:
        infer_fn = carregar_extrator_de_embeddings()
        calcular_embedding = lambda caminho: imagem_para_embedding(caminho, infer_fn)
    else:
        extrator = carregar_extrator_substituto()
        calcular_embedding = lambda caminho: imagem_para_embedding_substituto(caminho, extrator)

    model = tf.keras.models.load_model(caminho_modelo, compile=False)

    y_true, y_pred = [], []
    for i, (_, linha) in enumerate(test_df.iterrows(), start=1):
        embedding = calcular_embedding(linha["path"]).numpy()
        embedding = np.expand_dims(embedding, axis=0)
        predicao = model.predict(embedding, verbose=0)
        y_pred.append(int(np.argmax(predicao, axis=1)[0]))
        y_true.append(linha["label"])
        if i % 100 == 0 or i == len(test_df):
            print(f"  {i}/{len(test_df)} imagens avaliadas...")

    return np.array(y_true), np.array(y_pred)


def gerar_grafico(resultados, caminho_saida="comparacao_modelos.png"):
    """Gera e salva um gráfico de barras comparando accuracy e F1-macro dos modelos."""
    nomes = list(resultados.keys())
    acuracias = [resultados[n]["accuracy"] * 100 for n in nomes]
    f1s = [resultados[n]["f1_macro"] * 100 for n in nomes]

    x = np.arange(len(nomes))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    barras_acc = ax.bar(x - largura / 2, acuracias, largura, label="Acurácia", color="#4C72B0")
    barras_f1 = ax.bar(x + largura / 2, f1s, largura, label="F1-macro", color="#DD8452")

    ax.set_ylabel("Percentual (%)")
    ax.set_title("Comparação dos modelos — mesmo conjunto de teste reservado")
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(barras_acc, fmt="%.1f")
    ax.bar_label(barras_f1, fmt="%.1f")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    print(f"\nGráfico salvo em: {caminho_saida}")
    plt.show()


def main():
    print("Carregando o conjunto de teste compartilhado (mesmo para os três modelos)...")
    df = load_ham10000_data()
    _, _, test_df = criar_splits(df)
    print(f"Avaliando em {len(test_df)} imagens de teste, nunca usadas em nenhum treino.\n")

    avaliadores = {
        "EfficientNetV2": (avaliar_modelo_cnn, CAMINHOS_MODELOS["EfficientNetV2"]),
        "ConvNeXt": (avaliar_modelo_cnn, CAMINHOS_MODELOS["ConvNeXt"]),
        "Derm Foundation": (avaliar_derm_foundation, CAMINHOS_MODELOS["Derm Foundation"]),
    }

    resultados = {}
    for nome, (func_avaliar, caminho) in avaliadores.items():
        print(f"Avaliando {nome}...")
        saida = func_avaliar(caminho, test_df)
        if saida is None:
            print(f"  Arquivo não encontrado em '{caminho}' — pulando (treine esse modelo primeiro).\n")
            continue
        y_true, y_pred = saida
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        resultados[nome] = {"accuracy": acc, "f1_macro": f1_macro}
        print(f"  Acurácia: {acc:.2%} | F1-macro: {f1_macro:.2%}\n")

    if not resultados:
        print("Nenhum modelo treinado foi encontrado nos caminhos esperados. Treine ao menos um antes de comparar.")
        return

    melhor = max(resultados, key=lambda nome: resultados[nome]["f1_macro"])
    print(f">> Melhor modelo (por F1-macro — métrica mais justa com classes raras como melanoma): {melhor}")

    gerar_grafico(resultados)


if __name__ == "__main__":
    main()
