"""
dataset_ham10000.py — Módulo único e compartilhado para carregar o
HAM10000 e dividir treino/validação/teste.

Por que este módulo existe: antes, load_ham10000_data() estava
duplicada nos três modelos (EfficientNetV2, ConvNeXt, Derm Foundation).
Além do código repetido, isso significava que cada modelo podia acabar
usando um conjunto de teste diferente — tornando a comparação entre
eles injusta. Este módulo garante:

  1. Uma ÚNICA implementação de carregamento (corrige um bug uma vez,
     corrige nos três modelos).
  2. Um split de TESTE fixo (mesmo random_state, mesmas imagens) para
     os três modelos — permite comparar de forma justa.
  3. O conjunto de teste nunca é usado durante o treino (nem para
     EarlyStopping, nem para ModelCheckpoint) — só para avaliação final,
     em comparar_modelos.py.
"""

import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

CAMINHO_SPLIT_TESTE = "test_split_ham10000.csv"


def load_ham10000_data():
    """Busca automaticamente o arquivo HAM10000_metadata.csv em qualquer pasta do Kaggle."""
    csv_matches = glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)

    if not csv_matches:
        raise FileNotFoundError("Não foi possível encontrar HAM10000_metadata.csv em /kaggle/input/")

    metadata_path = csv_matches[0]
    base_dir = os.path.dirname(metadata_path)
    print(f"Dataset carregado com sucesso de: {base_dir}")

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


def criar_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Divide em treino/validação/teste, sempre com a MESMA semente —
    garante que os três modelos usem exatamente as mesmas imagens de
    teste, nunca vistas durante nenhum treino.

    O split de teste é salvo em disco (CAMINHO_SPLIT_TESTE) na primeira
    vez, e reaproveitado nas próximas — assim comparar_modelos.py pode
    carregar exatamente o mesmo conjunto depois, mesmo em outra sessão.
    """
    # Primeiro separa o teste (15%) do resto.
    resto_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    # Depois separa treino/validação dentro do que sobrou.
    val_fraction_do_resto = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        resto_df, test_size=val_fraction_do_resto, random_state=random_state, stratify=resto_df["label"]
    )

    if not os.path.exists(CAMINHO_SPLIT_TESTE):
        test_df[["image_id", "path", "dx", "label"]].to_csv(CAMINHO_SPLIT_TESTE, index=False)
        print(f"Split de teste salvo em: {CAMINHO_SPLIT_TESTE} ({len(test_df)} imagens)")

    # Não resetamos o índice aqui de propósito: train_df.index / val_df.index /
    # test_df.index continuam apontando para a posição original em df (que já
    # é 0..N-1, vindo de load_ham10000_data()). Isso permite que quem precisar
    # (como o Derm Foundation, que calcula embeddings para o df inteiro de uma
    # vez) fatie um array já calculado usando esses índices diretamente.
    return train_df, val_df, test_df
