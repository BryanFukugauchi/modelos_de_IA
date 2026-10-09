"""
pad_ufes.py — Avaliação e treino no PAD-UFES-20 (fotos clínicas de celular),
reaproveitando a pipeline do Derm Foundation (train.py / model.py).

Dois modos:
  --modo externo : carrega o modelo treinado no HAM10000 (derm_foundation_model.keras)
                   e mede o desempenho no PAD-UFES-20 SEM retreinar (domain shift).
  --modo treino  : treina uma cabeça nova só com o PAD-UFES-20, com validação cruzada
                   de 5 folds agrupada por paciente (sem vazamento).
  --modo ambos   : roda os dois.

Configuração no .env:
  PAD_PATH=/caminho/para/PAD-UFES-20   (pasta com metadata.csv e imgs_part_1/2/3)

Exemplos:
  python pad_ufes.py --modo externo
  python pad_ufes.py --modo treino
  python pad_ufes.py --modo treino --sem_meta
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model, carregar_extrator_de_embeddings
from train import preparar_exemplo_png  # mesmo pré-processamento do HAM10000

PAD_PATH = os.getenv("PAD_PATH", ".")
PAD_CACHE = "pad_embeddings_cache.npz"

# ---------------------------------------------------------------------------
# Mapeamentos PAD-UFES-20 -> HAM10000
# Códigos do HAM (ordem alfabética do pd.Categorical):
#   0 akiec | 1 bcc | 2 bkl | 3 df | 4 mel | 5 nv | 6 vasc
# ATENÇÃO: é uma aproximação. SCC invasivo não é exatamente "akiec"
# (que no HAM inclui doença de Bowen/carcinoma intraepitelial). Documente isso no TCC.
# ---------------------------------------------------------------------------
PAD_PARA_HAM = {"ACK": 0, "SCC": 0, "BCC": 1, "SEK": 2, "MEL": 4, "NEV": 5}
HAM_NOMES = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
MEL_HAM = 4

# Colunas de localização do HAM10000 (mesma ordem alfabética do get_dummies do treino)
HAM_LOCALIZACOES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity",
]
REGIAO_PAD_PARA_HAM = {
    "ABDOMEN": "abdomen", "BACK": "back", "CHEST": "chest", "EAR": "ear",
    "FACE": "face", "LIP": "face", "NOSE": "face", "FOOT": "foot",
    "HAND": "hand", "NECK": "neck", "SCALP": "scalp",
    "ARM": "upper extremity", "FOREARM": "upper extremity",
    "THIGH": "lower extremity",
}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_pad(raiz: str = PAD_PATH) -> pd.DataFrame:
    csv_path = os.path.join(raiz, "metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} não encontrado. Defina PAD_PATH no .env apontando para a "
            f"pasta do PAD-UFES-20 (que contém metadata.csv)."
        )

    df = pd.read_csv(csv_path)

    # Localiza as imagens em qualquer subpasta (imgs_part_1, imgs_part_2, ...)
    caminhos = {}
    for pasta, _, arquivos in os.walk(raiz):
        for arq in arquivos:
            if arq.lower().endswith((".png", ".jpg", ".jpeg")):
                caminhos[arq] = os.path.join(pasta, arq)

    df["path"] = df["img_id"].map(caminhos)
    faltando = int(df["path"].isna().sum())
    if faltando:
        raise FileNotFoundError(f"{faltando} imagens do metadata.csv não foram encontradas em {raiz}.")

    df["diagnostic"] = df["diagnostic"].astype(str).str.upper()
    df = df.reset_index(drop=True)  # garante alinhamento linha <-> embedding
    print(f"PAD-UFES-20: {len(df)} imagens, {df['patient_id'].nunique()} pacientes.")
    print(df["diagnostic"].value_counts().to_string())
    return df


def calcular_ou_carregar_embeddings(df: pd.DataFrame) -> np.ndarray:
    ids = df["img_id"].values.astype(str)

    if os.path.exists(PAD_CACHE):
        cache = np.load(PAD_CACHE)
        if "ids" in cache.files and np.array_equal(cache["ids"], ids):
            print(f">> Reaproveitando cache: {PAD_CACHE}")
            return cache["embeddings"]
        print(">> Cache do PAD diferente do dataset atual — recalculando...")

    print(">> Extraindo embeddings do Derm Foundation (CPU)...")
    infer_fn = carregar_extrator_de_embeddings()
    caminhos = df["path"].tolist()
    total, batch_size = len(caminhos), 32
    partes = []

    for i in range(0, total, batch_size):
        lote = caminhos[i : i + batch_size]
        with ThreadPoolExecutor() as ex:
            exemplos = list(ex.map(preparar_exemplo_png, lote))
        with tf.device("/CPU:0"):
            saida = infer_fn(inputs=tf.constant(exemplos))
        partes.append(saida["embedding"].numpy())
        print(f"  {min(i + batch_size, total)}/{total} embeddings calculados...")

    embeddings = np.vstack(partes)
    np.savez(PAD_CACHE, embeddings=embeddings, ids=ids)
    print(f">> Embeddings salvos em {PAD_CACHE}")
    return embeddings


# ---------------------------------------------------------------------------
# Metadados
# ---------------------------------------------------------------------------
def meta_formato_ham(df: pd.DataFrame) -> np.ndarray:
    """
    Reproduz EXATAMENTE as 18 colunas de metadados usadas no treino do HAM10000:
    [age/100, sex_male, sex_unknown, 15 localizações em ordem alfabética].
    """
    idade = pd.to_numeric(df["age"], errors="coerce")
    idade = idade.fillna(idade.median()) / 100.0

    sexo = df["gender"].astype(str).str.upper()
    meta = pd.DataFrame(
        {
            "age": idade.values,
            "sex_male": (sexo == "MALE").astype(float).values,
            "sex_unknown": (~sexo.isin(["MALE", "FEMALE"])).astype(float).values,
        }
    )

    loc = df["region"].astype(str).str.upper().map(REGIAO_PAD_PARA_HAM).fillna("unknown")
    for nome in HAM_LOCALIZACOES:
        meta[nome] = (loc.values == nome).astype(float)

    return meta.values.astype("float32")


def meta_proprio(df: pd.DataFrame) -> np.ndarray:
    """Metadados no formato do próprio PAD-UFES-20 (para treinar uma cabeça nova)."""
    idade = pd.to_numeric(df["age"], errors="coerce")
    meta = pd.DataFrame({"age": idade.fillna(idade.median()).values / 100.0})
    sexo = df["gender"].astype(str).str.upper()
    meta["sex_male"] = (sexo == "MALE").astype(float).values
    meta["sex_unknown"] = (~sexo.isin(["MALE", "FEMALE"])).astype(float).values
    regiao = pd.get_dummies(df["region"].astype(str).str.upper(), prefix="reg", dtype=float)
    meta = pd.concat([meta.reset_index(drop=True), regiao.reset_index(drop=True)], axis=1)
    return meta.values.astype("float32")


# ---------------------------------------------------------------------------
# Modo 1: avaliação externa do modelo treinado no HAM10000
# ---------------------------------------------------------------------------
def avaliar_externo(df, emb, args):
    print("\n================ AVALIAÇÃO EXTERNA (modelo HAM10000 -> PAD-UFES-20) ================")
    modelo = tf.keras.models.load_model(args.modelo)

    X = np.hstack([emb, meta_formato_ham(df)])
    esperado = modelo.input_shape[1]
    assert X.shape[1] == esperado, (
        f"O modelo espera {esperado} entradas, mas montei {X.shape[1]}. "
        f"Treinou com metadados? Se treinou só com embeddings (6144), use embeddings sem metadados."
    )

    y = df["diagnostic"].map(PAD_PARA_HAM).values.astype(int)
    probs = modelo.predict(X, verbose=0)

    fator = np.ones(probs.shape[1])
    fator[MEL_HAM] = args.fator_mel
    pred = (probs * fator).argmax(axis=1)

    labels = [0, 1, 2, 4, 5]  # classes que existem no PAD (df e vasc não existem)
    nomes = ["akiec (ACK+SCC)", "bcc", "bkl (SEK)", "mel", "nv"]
    print(f"Fator de melanoma: {args.fator_mel}")
    print(classification_report(y, pred, labels=labels, target_names=nomes, zero_division=0))
    print("Balanced acc:", round(balanced_accuracy_score(y, pred), 4))
    print("Macro-F1 (classes do PAD):", round(f1_score(y, pred, labels=labels, average="macro", zero_division=0), 4))

    print("\nMatriz de confusão (linhas = real, colunas = predito; ordem akiec,bcc,bkl,df,mel,nv,vasc):")
    print(confusion_matrix(y, pred, labels=list(range(7))))

    frac_inexistentes = np.isin(pred, [3, 6]).mean()
    print(f"\nPredições em classes que não existem no PAD (df/vasc): {frac_inexistentes:.1%}")


# ---------------------------------------------------------------------------
# Modo 2: treino no PAD-UFES-20 com validação cruzada agrupada por paciente
# ---------------------------------------------------------------------------
def treinar_cv(df, emb, args):
    print("\n================ TREINO NO PAD-UFES-20 (5-fold, agrupado por paciente) ================")
    classes = sorted(df["diagnostic"].unique())
    mapa = {c: i for i, c in enumerate(classes)}
    y = df["diagnostic"].map(mapa).values.astype(int)

    X = emb if args.sem_meta else np.hstack([emb, meta_proprio(df)])
    print(f"Classes: {classes} | entradas: {X.shape[1]} | metadados: {'não' if args.sem_meta else 'sim'}")

    n_folds = 5
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = [idx for _, idx in sgkf.split(X, y, groups=df["patient_id"].values)]

    oof = np.zeros(len(y), dtype=int)
    baccs, f1s = [], []

    for k in range(n_folds):
        teste = folds[k]
        val = folds[(k + 1) % n_folds]
        treino = np.concatenate([folds[j] for j in range(n_folds) if j not in (k, (k + 1) % n_folds)])

        cls_tr = np.unique(y[treino])
        pesos = np.sqrt(compute_class_weight("balanced", classes=cls_tr, y=y[treino]))
        class_weight = dict(zip(cls_tr.tolist(), pesos.tolist()))

        tf.keras.utils.set_random_seed(42 + k)
        modelo = build_model(num_classes=len(classes), embedding_dim=X.shape[1], X_adapt=X[treino])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        modelo.fit(
            X[treino], y[treino],
            validation_data=(X[val], y[val]),
            epochs=args.epochs,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
            ],
            verbose=0,
        )

        pred = modelo.predict(X[teste], verbose=0).argmax(axis=1)
        oof[teste] = pred
        bacc = balanced_accuracy_score(y[teste], pred)
        f1 = f1_score(y[teste], pred, average="macro", zero_division=0)
        baccs.append(bacc)
        f1s.append(f1)
        print(f"Fold {k + 1}/{n_folds}: balanced acc = {bacc:.3f} | macro-F1 = {f1:.3f} "
              f"| treino={len(treino)} val={len(val)} teste={len(teste)}")

    print(f"\nBalanced acc: {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")
    print(f"Macro-F1:     {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    print("\nRelatório agregado (predições fora-da-amostra de todos os folds):")
    print(classification_report(y, oof, target_names=classes, zero_division=0))
    print("Matriz de confusão (ordem:", classes, ")")
    print(confusion_matrix(y, oof))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação/treino no PAD-UFES-20")
    parser.add_argument("--modo", choices=["externo", "treino", "ambos"], default="externo")
    parser.add_argument("--modelo", default="derm_foundation_model.keras", help="modelo treinado no HAM10000")
    parser.add_argument("--fator_mel", type=float, default=2.0, help="reforço da probabilidade de melanoma")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--sem_meta", action="store_true", help="modo treino: só embeddings, sem metadados")
    args = parser.parse_args()

    df = carregar_pad()
    emb = calcular_ou_carregar_embeddings(df)

    if args.modo in ("externo", "ambos"):
        avaliar_externo(df, emb, args)
    if args.modo in ("treino", "ambos"):
        treinar_cv(df, emb, args)
