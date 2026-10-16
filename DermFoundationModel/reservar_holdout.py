"""
reservar_holdout.py — Etapa 0: reserva os pacientes do teste final do PAD-UFES-20.

Separa uma fração dos PACIENTES (não das imagens), estratificada pelo diagnóstico,
e salva os patient_id em holdout_pacientes.csv. Esses pacientes ficam de fora de
toda validação cruzada, calibração e escolha de limiar, e só são usados uma vez,
na avaliação final (Etapa 6).

Rode UMA vez e versione o CSV no Git:
  python reservar_holdout.py              # reserva ~20%
  python reservar_holdout.py --fracao 0.15

O script se recusa a sobrescrever um holdout existente (use --forcar só se tiver
certeza de que nada foi avaliado com o holdout atual).
"""

import argparse
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedGroupKFold

load_dotenv()
PAD_PATH = os.getenv("PAD_PATH", ".")
ARQUIVO_HOLDOUT = "holdout_pacientes.csv"


def main(fracao: float, seed: int, forcar: bool):
    if os.path.exists(ARQUIVO_HOLDOUT) and not forcar:
        raise SystemExit(
            f"{ARQUIVO_HOLDOUT} já existe. O holdout deve ser criado uma única vez. "
            f"Use --forcar apenas se tiver certeza."
        )

    csv_path = os.path.join(PAD_PATH, "metadata.csv")
    df = pd.read_csv(csv_path)
    df["diagnostic"] = df["diagnostic"].astype(str).str.upper()

    # Ex.: fracao 0.20 -> 5 partes; 0.15 -> 7 partes (~14%). Pega a primeira parte.
    n_partes = max(2, round(1 / fracao))
    sgkf = StratifiedGroupKFold(n_splits=n_partes, shuffle=True, random_state=seed)
    _, idx_holdout = next(sgkf.split(df, df["diagnostic"], groups=df["patient_id"]))

    pacientes_holdout = np.sort(df.iloc[idx_holdout]["patient_id"].unique())
    em_holdout = df["patient_id"].isin(pacientes_holdout)

    # Garantias
    dev, hold = df[~em_holdout], df[em_holdout]
    assert not set(dev["patient_id"]) & set(hold["patient_id"]), "Paciente nos dois conjuntos!"
    faltando = set(df["diagnostic"]) - set(hold["diagnostic"])
    if faltando:
        raise SystemExit(f"Classes ausentes no holdout: {faltando}. Tente outra --seed.")

    # Resumo
    print(f"Partes: {n_partes} | fração real: {len(hold) / len(df):.1%} das imagens, "
          f"{hold['patient_id'].nunique() / df['patient_id'].nunique():.1%} dos pacientes\n")
    resumo = pd.DataFrame({
        "desenvolvimento": dev["diagnostic"].value_counts(),
        "holdout": hold["diagnostic"].value_counts(),
    }).fillna(0).astype(int)
    resumo["% holdout"] = (resumo["holdout"] / (resumo["holdout"] + resumo["desenvolvimento"]) * 100).round(1)
    print(resumo.to_string())
    print(f"\nTotal: desenvolvimento={len(dev)} imagens / holdout={len(hold)} imagens")

    pd.DataFrame({"patient_id": pacientes_holdout}).to_csv(ARQUIVO_HOLDOUT, index=False)
    print(f"\n>> {len(pacientes_holdout)} pacientes salvos em {ARQUIVO_HOLDOUT} (seed={seed}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reserva pacientes do teste final (PAD-UFES-20)")
    parser.add_argument("--fracao", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--forcar", action="store_true")
    args = parser.parse_args()
    main(args.fracao, args.seed, args.forcar)
