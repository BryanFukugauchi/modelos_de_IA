"""
listar_erros.py — Análise qualitativa dos erros graves da triagem.

Reproduz EXATAMENTE os níveis de prioridade da validação cruzada do
avaliar_triagem.py (mesmos folds, mesmos alvos lidos de config_triagem.json)
e separa os erros que mais importam:

  - cânceres (MEL, BCC, SCC) que caíram em ROTINA ou ACOMPANHAMENTO
  - melanomas que não chegaram à ALTA

Usa só os dados de DESENVOLVIMENTO: o conjunto reservado continua intocado.

Saídas:
  erros_triagem.csv          lista dos casos, com probabilidades e metadados
  erros_triagem.png          prancha com as fotos (legenda: real, nível, previsão)
  erros_triagem/             cópia das fotos, com o diagnóstico e o nível no nome

Rode depois do avaliar_triagem.py:
  python listar_erros.py
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaliar_triagem import (  # noqa: E402
    ACOMP, ALTA, CANCER, NIVEIS,
    ajustar_calibrador, aplicar_calibrador, atribuir_niveis, escolher_limiares,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PAD_PATH = os.getenv("PAD_PATH", ".")
PASTA_SAIDA = "erros_triagem"


def niveis_validacao_cruzada(probs, y, grupos, classes, alvos):
    """Mesmo procedimento do avaliar_triagem.py (mesma seed dos folds)."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    pcal = np.zeros_like(probs)
    niv = np.zeros(len(y), dtype=int)
    for tr, te in sgkf.split(probs, y, groups=grupos):
        T, b = ajustar_calibrador(probs[tr], y[tr])
        p_tr, p_te = aplicar_calibrador(probs[tr], T, b), aplicar_calibrador(probs[te], T, b)
        lim = escolher_limiares(p_tr, y[tr], classes, alvos)
        niv[te] = atribuir_niveis(p_te, classes, lim)
        pcal[te] = p_te
    return pcal, niv


def localizar_imagens(raiz):
    caminhos = {}
    for pasta, _, arquivos in os.walk(raiz):
        for arq in arquivos:
            if arq.lower().endswith((".png", ".jpg", ".jpeg")):
                caminhos[arq] = os.path.join(pasta, arq)
    return caminhos


def montar_prancha(erros, arquivo, colunas=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print(">> matplotlib/Pillow não instalados; prancha não gerada.")
        return

    com_foto = erros[erros["path"].notna()]
    if com_foto.empty:
        print(">> Nenhuma foto encontrada; prancha não gerada.")
        return

    linhas = int(np.ceil(len(com_foto) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 2.6, linhas * 3.1))
    eixos = np.atleast_1d(eixos).ravel()
    for ax in eixos:
        ax.axis("off")

    for ax, (_, r) in zip(eixos, com_foto.iterrows()):
        img = Image.open(r["path"]).convert("RGB")
        img.thumbnail((400, 400))
        ax.imshow(img)
        cor = "#b91c1c" if r["real"] == "MEL" else "#1f2937"
        ax.set_title(
            f"Real: {r['real']} → {r['nivel']}\n"
            f"Previu {r['mais_provavel']} ({r['p_mais_provavel']:.0%})\n"
            f"susp. câncer {r['suspeita_cancer']:.0%}",
            fontsize=8, color=cor,
        )
        ax.text(0.5, -0.04, r["img_id"], transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color="#6b7280")

    fig.suptitle("Erros graves da triagem (validação cruzada, dados de desenvolvimento)", fontsize=11)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=130)
    plt.close(fig)
    print(f">> Prancha salva em {arquivo}")


def main():
    with open("config_triagem.json", encoding="utf-8") as f:
        config = json.load(f)
    dados = np.load(config["base"], allow_pickle=True)

    classes = [str(c) for c in dados["classes"]]
    y = dados["y"].astype(int)
    grupos = dados["patient_id"]
    img_ids = [str(i) for i in dados["img_id"]]
    probs = dados["probs"].mean(axis=0)
    alvos = config["sensibilidades_alvo"]

    pcal, niv = niveis_validacao_cruzada(probs, y, grupos, classes, alvos)

    ic = [classes.index(c) for c in CANCER]
    mel = classes.index("MEL")
    e_cancer = np.isin(y, ic)
    perdido = e_cancer & (niv <= ACOMP)
    mel_fora_alta = (y == mel) & (niv != ALTA)
    selecionado = perdido | mel_fora_alta

    tab = pd.DataFrame({
        "img_id": img_ids,
        "patient_id": grupos,
        "real": [classes[i] for i in y],
        "nivel": [NIVEIS[i] for i in niv],
        "mais_provavel": [classes[i] for i in pcal.argmax(axis=1)],
        "p_mais_provavel": pcal.max(axis=1),
        "suspeita_cancer": pcal[:, ic].sum(axis=1),
        **{f"p_{c}": pcal[:, i] for i, c in enumerate(classes)},
    })[selecionado].copy()

    # Metadados úteis para a discussão (região, idade, sintomas, dados do exame)
    csv_meta = os.path.join(PAD_PATH, "metadata.csv")
    if os.path.exists(csv_meta):
        meta = pd.read_csv(csv_meta)
        colunas = [c for c in ["img_id", "age", "region", "diameter_1", "diameter_2", "fitspatrick",
                               "grew", "changed", "bleed", "itch", "hurt", "elevation"] if c in meta.columns]
        tab = tab.merge(meta[colunas], on="img_id", how="left")

    caminhos = localizar_imagens(PAD_PATH)
    tab["path"] = tab["img_id"].map(caminhos)

    ordem_real = {"MEL": 0, "SCC": 1, "BCC": 2}
    tab = tab.sort_values(
        ["real", "suspeita_cancer"], key=lambda s: s.map(ordem_real) if s.name == "real" else s
    ).reset_index(drop=True)
    tab.drop(columns=["path"]).to_csv("erros_triagem.csv", index=False)

    print(f"Alvos usados: {alvos}")
    print(f"Cânceres perdidos (ROTINA/ACOMP.): {perdido.sum()} de {e_cancer.sum()}")
    print(f"Melanomas fora da ALTA: {mel_fora_alta.sum()} de {(y == mel).sum()}")
    print("\nPor diagnóstico real e nível:")
    print(pd.crosstab(tab["real"], tab["nivel"]).to_string())
    print("\nPara onde o modelo os levou (classe mais provável):")
    print(pd.crosstab(tab["real"], tab["mais_provavel"]).to_string())

    cols_console = ["img_id", "real", "nivel", "mais_provavel", "p_mais_provavel", "suspeita_cancer"]
    cols_console += [c for c in ["age", "region", "fitspatrick", "grew", "bleed"] if c in tab.columns]
    print("\nCasos:")
    print(tab[cols_console].to_string(index=False, formatters={
        "p_mais_provavel": "{:.0%}".format, "suspeita_cancer": "{:.0%}".format}))
    print("\n>> Lista completa salva em erros_triagem.csv")

    faltando = tab["path"].isna().sum()
    if faltando:
        print(f">> AVISO: {faltando} fotos não encontradas em PAD_PATH={PAD_PATH}.")

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    for _, r in tab[tab["path"].notna()].iterrows():
        destino = os.path.join(PASTA_SAIDA, f"{r['real']}_{r['nivel']}_{r['img_id']}")
        shutil.copy2(r["path"], destino)
    print(f">> Fotos copiadas para a pasta {PASTA_SAIDA}/")

    montar_prancha(tab, "erros_triagem.png")


if __name__ == "__main__":
    main()
