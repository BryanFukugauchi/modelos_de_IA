"""
pad_ufes.py — Avaliação e treino no PAD-UFES-20 (fotos clínicas de celular),
reaproveitando a pipeline do Derm Foundation (train.py / model.py).

Dois modos:
  --modo externo : carrega o modelo treinado no HAM10000 (derm_foundation_model.keras)
                   e mede o desempenho no PAD-UFES-20 SEM retreinar (domain shift).
  --modo treino  : treina uma cabeça nova só com o PAD-UFES-20, com validação cruzada
                   de 5 folds agrupada por paciente, repetida com várias seeds.
  --modo ambos   : roda os dois.

Os pacientes listados em holdout_pacientes.csv (Etapa 0) são SEMPRE removidos
antes de qualquer experimento. Eles só serão usados na avaliação final.

Configuração no .env:
  PAD_PATH=/caminho/para/PAD-UFES-20   (pasta com metadata.csv e imgs_part_1/2/3)

Exemplos:
  python pad_ufes.py --modo treino                      # 4 configurações x 5 seeds
  python pad_ufes.py --modo treino --meta sintomas      # só uma configuração
  python pad_ufes.py --modo treino --seeds 3            # mais rápido
  python pad_ufes.py --modo externo
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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model, carregar_extrator_de_embeddings
from train import preparar_exemplo_png  # mesmo pré-processamento do HAM10000

PAD_PATH = os.getenv("PAD_PATH", ".")
PAD_CACHE = "pad_embeddings_cache.npz"
ARQUIVO_HOLDOUT = "holdout_pacientes.csv"

# ---------------------------------------------------------------------------
# Mapeamentos PAD-UFES-20 -> HAM10000 (usados só no modo externo)
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
# Metadados do próprio PAD (modo treino)
# ---------------------------------------------------------------------------
REGIOES_PAD = ["ABDOMEN", "ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT",
               "FOREARM", "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH"]

# Campos com "não sei" quase inexistente (2-10 casos): seguros.
SINTOMAS = ["bleed", "itch", "hurt", "elevation"]
# "Não sei" concentrado em BCC/SCC: só no experimento 'evolucao', com "não sei" = "não".
SINTOMAS_EVOLUCAO = ["grew", "changed"]
# FORA do modelo: gender, skin_cancer_history, cancer_history
# (ausentes em 804 registros, todos de lesões benignas -> vazamento).

CONFIGS_META = ["nenhum", "basico", "sintomas", "evolucao"]
CANCER = ["BCC", "MEL", "SCC"]


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
    # dtype="U" garante um array de texto comum (no pandas novo, .values.astype(str)
    # pode virar array de objetos, que o numpy só carrega com pickle).
    ids = np.array(df["img_id"].astype(str).tolist(), dtype="U")

    if os.path.exists(PAD_CACHE):
        # allow_pickle=True é seguro aqui: o arquivo foi gerado por este próprio script.
        cache = np.load(PAD_CACHE, allow_pickle=True)
        if "ids" in cache.files and [str(x) for x in cache["ids"]] == ids.tolist():
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


def remover_holdout(df: pd.DataFrame, emb: np.ndarray):
    """Etapa 0: remove os pacientes reservados para o teste final."""
    if not os.path.exists(ARQUIVO_HOLDOUT):
        raise FileNotFoundError(
            f"{ARQUIVO_HOLDOUT} não encontrado. Rode antes: python reservar_holdout.py"
        )
    holdout = set(pd.read_csv(ARQUIVO_HOLDOUT)["patient_id"])
    dev = ~df["patient_id"].isin(holdout).values
    print(f">> Holdout removido: {(~dev).sum()} imagens. Restam {dev.sum()} para desenvolvimento.")
    return df[dev].reset_index(drop=True), emb[dev]


# ---------------------------------------------------------------------------
# Metadados
# ---------------------------------------------------------------------------
def meta_formato_ham(df: pd.DataFrame) -> np.ndarray:
    """
    Reproduz EXATAMENTE as 18 colunas de metadados usadas no treino do HAM10000:
    [age/100, sex_male, sex_unknown, 15 localizações em ordem alfabética].
    Usado só no modo externo (modelo treinado no HAM, não no PAD).
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


def _sim_nao_desconhecido(serie: pd.Series) -> pd.Series:
    s = serie.astype(str).str.strip().str.upper()
    return s.where(s.isin(["TRUE", "FALSE"]), "UNK")


def meta_proprio(df: pd.DataFrame, nivel: str = "sintomas") -> np.ndarray:
    """
    nivel='basico'   -> idade + região
    nivel='sintomas' -> + sangramento, coceira, dor, elevação
    nivel='evolucao' -> + crescimento, mudança
    Colunas fixas e sem estatísticas dos dados (sem vazamento entre folds).
    """
    idade = pd.to_numeric(df["age"], errors="coerce")
    meta = pd.DataFrame({"age": (idade.fillna(0) / 100.0).values})

    regiao = df["region"].astype(str).str.upper()
    for r in REGIOES_PAD:
        meta[f"reg_{r}"] = (regiao == r).astype(float).values
    meta["reg_OUTRA"] = (~regiao.isin(REGIOES_PAD)).astype(float).values

    campos = []
    if nivel in ("sintomas", "evolucao"):
        campos += SINTOMAS
    if nivel == "evolucao":
        campos += SINTOMAS_EVOLUCAO
    for campo in campos:
        meta[f"{campo}_sim"] = (_sim_nao_desconhecido(df[campo]) == "TRUE").astype(float).values

    return meta.values.astype("float32")


def montar_X(df: pd.DataFrame, emb: np.ndarray, meta: str) -> np.ndarray:
    if meta == "nenhum":
        return emb
    return np.hstack([emb, meta_proprio(df, nivel=meta)])


# ---------------------------------------------------------------------------
# Modo 1: avaliação externa do modelo treinado no HAM10000
# ---------------------------------------------------------------------------
def avaliar_externo(df, emb, args):
    print("\n================ AVALIAÇÃO EXTERNA (modelo HAM10000 -> PAD-UFES-20) ================")
    modelo = tf.keras.models.load_model(args.modelo)

    esperado = modelo.input_shape[1]
    if esperado == emb.shape[1]:
        X = emb
        print(f"Modelo com {esperado} entradas: treinado SÓ com embeddings (sem metadados).")
    else:
        X = np.hstack([emb, meta_formato_ham(df)])
        print(f"Modelo com {esperado} entradas: embeddings + metadados.")
    assert X.shape[1] == esperado, (
        f"O modelo espera {esperado} entradas, mas montei {X.shape[1]}. "
        f"O formato dos metadados no treino do HAM10000 foi diferente do que o script reproduz."
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
# Modo 2: treino no PAD-UFES-20, 5 folds por paciente, repetido com várias seeds
# ---------------------------------------------------------------------------
def oof_uma_seed(X, y, grupos, n_classes, seed, epochs):
    """
    Validação cruzada de 5 folds agrupada por paciente. Devolve as probabilidades
    fora da amostra (cada imagem prevista pelo modelo que NÃO a viu no treino).

    A divisão depende só de y, dos pacientes e da seed — então, para a mesma seed,
    todas as configurações de metadados usam exatamente os mesmos folds
    (comparação pareada e justa).
    """
    n_folds = 5
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [idx for _, idx in sgkf.split(X, y, groups=grupos)]
    probs = np.zeros((len(y), n_classes), dtype="float32")

    for k in range(n_folds):
        teste = folds[k]
        val = folds[(k + 1) % n_folds]
        treino = np.concatenate([folds[j] for j in range(n_folds) if j not in (k, (k + 1) % n_folds)])

        cls_tr = np.unique(y[treino])
        pesos = np.sqrt(compute_class_weight("balanced", classes=cls_tr, y=y[treino]))
        class_weight = dict(zip(cls_tr.tolist(), pesos.tolist()))

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + k)  # seed 42 reproduz os resultados anteriores
        modelo = build_model(num_classes=n_classes, embedding_dim=X.shape[1], X_adapt=X[treino])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        modelo.fit(
            X[treino], y[treino],
            validation_data=(X[val], y[val]),
            epochs=epochs,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
            ],
            verbose=0,
        )
        probs[teste] = modelo.predict(X[teste], verbose=0)

    return probs


def calcular_metricas(y, probs, classes):
    """Métricas multiclasse + câncer (MEL+BCC+SCC) vs. não câncer."""
    pred = probs.argmax(axis=1)
    idx_cancer = [classes.index(c) for c in CANCER]
    mel = classes.index("MEL")

    e_cancer = np.isin(y, idx_cancer)
    pred_cancer = np.isin(pred, idx_cancer)
    pontuacao = probs[:, idx_cancer].sum(axis=1)  # pontuação de suspeita (ainda sem calibrar)

    return {
        "bacc": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
        "acc": float((pred == y).mean()),
        "auc_cancer": roc_auc_score(e_cancer, pontuacao),
        "sens_cancer": float(pred_cancer[e_cancer].mean()),
        "espec": float((~pred_cancer[~e_cancer]).mean()),
        "recall_mel": float((pred[y == mel] == mel).mean()),
    }


NOMES_METRICAS = {
    "bacc": "Balanced acc",
    "f1": "Macro-F1",
    "acc": "Acurácia",
    "auc_cancer": "AUC câncer",
    "sens_cancer": "Sensib. câncer",
    "espec": "Especif.",
    "recall_mel": "Recall MEL",
}


def _media_dp(valores):
    v = np.asarray(valores, dtype=float)
    dp = v.std(ddof=1) if len(v) > 1 else 0.0
    return v.mean(), dp


def treinar_cv(df, emb, args):
    print("\n================ TREINO NO PAD-UFES-20 (5-fold por paciente, várias seeds) ================")
    classes = sorted(df["diagnostic"].unique())
    mapa = {c: i for i, c in enumerate(classes)}
    y = df["diagnostic"].map(mapa).values.astype(int)
    grupos = df["patient_id"].values

    configs = CONFIGS_META if args.meta == "todos" else [args.meta]
    seeds = [42 + i for i in range(args.seeds)]
    print(f"Classes: {classes} | configurações: {configs} | seeds: {seeds}")
    print(f"Total de treinos: {len(configs)} x {len(seeds)} seeds x 5 folds = {len(configs) * len(seeds) * 5}")

    resultados = {}
    for meta in configs:
        X = montar_X(df, emb, meta)
        print(f"\n--- metadados: {meta} | entradas: {X.shape[1]} ---")

        por_seed, probs_seeds = [], []
        for s in seeds:
            probs = oof_uma_seed(X, y, grupos, len(classes), s, args.epochs)
            m = calcular_metricas(y, probs, classes)
            por_seed.append(m)
            probs_seeds.append(probs)
            print(f"  seed {s}: bacc={m['bacc']:.3f} | F1={m['f1']:.3f} | "
                  f"AUC câncer={m['auc_cancer']:.3f} | sens={m['sens_cancer']:.3f} | "
                  f"espec={m['espec']:.3f} | MEL={m['recall_mel']:.2f}")

        resultados[meta] = por_seed

        # Probabilidades fora da amostra: serão usadas na Etapa 3 (calibração e limiar).
        np.savez(
            f"oof_{meta}.npz",
            probs=np.stack(probs_seeds),          # (n_seeds, n_imagens, n_classes)
            y=y,
            seeds=np.array(seeds),
            classes=np.array(classes, dtype="U"),
            patient_id=np.array(grupos),
            img_id=np.array(df["img_id"].astype(str).tolist(), dtype="U"),
            entradas=X.shape[1],
        )
        print(f"  >> Probabilidades fora da amostra salvas em oof_{meta}.npz")

        if len(configs) == 1:
            pred = probs_seeds[0].argmax(axis=1)
            print(f"\nRelatório da seed {seeds[0]} (predições fora da amostra):")
            print(classification_report(y, pred, target_names=classes, zero_division=0))
            print("Matriz de confusão (ordem:", classes, ")")
            print(confusion_matrix(y, pred))

    # ---- Tabela-resumo: média ± desvio padrão entre seeds ----
    chaves = list(NOMES_METRICAS)
    linhas = []
    print(f"\n================ RESUMO ({len(seeds)} seeds; média ± desvio padrão) ================")
    print("Métricas calculadas sobre as predições fora da amostra de cada seed (1838 imagens).")
    print("Sensib./especif. usam o limiar padrão (classe mais provável); a AUC não depende de limiar.\n")
    cab = f"{'metadados':<10} | " + " | ".join(f"{NOMES_METRICAS[k]:>15}" for k in chaves)
    print(cab)
    print("-" * len(cab))
    for meta in configs:
        celulas = []
        linha_csv = {"metadados": meta, "entradas": montar_X(df.head(1), emb[:1], meta).shape[1]}
        for k in chaves:
            media, dp = _media_dp([r[k] for r in resultados[meta]])
            celulas.append(f"{media:.3f} ± {dp:.3f}")
            linha_csv[f"{k}_media"] = round(media, 4)
            linha_csv[f"{k}_dp"] = round(dp, 4)
        linhas.append(linha_csv)
        print(f"{meta:<10} | " + " | ".join(f"{c:>15}" for c in celulas))

    pd.DataFrame(linhas).to_csv("resultados_etapa2.csv", index=False)
    print("\n>> Tabela salva em resultados_etapa2.csv")

    # ---- Comparação pareada contra 'só imagem' (mesmos folds em cada seed) ----
    if "nenhum" in resultados and len(resultados) > 1:
        print("\nDiferença em relação a 'nenhum' (só imagem), seed a seed:")
        for k in ["auc_cancer", "bacc"]:
            base = np.array([r[k] for r in resultados["nenhum"]])
            for meta in configs:
                if meta == "nenhum":
                    continue
                d = np.array([r[k] for r in resultados[meta]]) - base
                media, dp = _media_dp(d)
                print(f"  {NOMES_METRICAS[k]:<13} {meta:<9}: {media:+.3f} ± {dp:.3f} "
                      f"| melhor em {(d > 0).sum()}/{len(d)} seeds")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação/treino no PAD-UFES-20")
    parser.add_argument("--modo", choices=["externo", "treino", "ambos"], default="externo")
    parser.add_argument("--modelo", default="derm_foundation_model.keras", help="modelo treinado no HAM10000")
    parser.add_argument("--fator_mel", type=float, default=2.0, help="reforço da probabilidade de melanoma")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--meta", choices=CONFIGS_META + ["todos"], default="todos",
                        help="modo treino: nenhum=só imagem | basico=idade+região | "
                             "sintomas=+sangramento,coceira,dor,elevação | "
                             "evolucao=+crescimento,mudança | todos=os quatro")
    parser.add_argument("--seeds", type=int, default=5, help="modo treino: quantas repetições da validação cruzada")
    args = parser.parse_args()

    df = carregar_pad()
    emb = calcular_ou_carregar_embeddings(df)  # cache continua com todas as imagens
    df, emb = remover_holdout(df, emb)

    if args.modo in ("externo", "ambos"):
        avaliar_externo(df, emb, args)
    if args.modo in ("treino", "ambos"):
        treinar_cv(df, emb, args)
