"""
avaliar_triagem.py — Etapa 3: pontuação de suspeita de câncer e níveis de prioridade.

NÃO treina nada. Usa as probabilidades fora da amostra salvas pelo pad_ufes.py
(oof_sintomas.npz) e faz três coisas:

  1. CALIBRAÇÃO: ajusta as probabilidades para que a pontuação de suspeita
     = P(MEL) + P(BCC) + P(SCC) reflita a frequência real de câncer no PAD-UFES-20.
  2. NÍVEIS DE PRIORIDADE POR GRAVIDADE (vale sempre o mais grave atingido):
        ALTA            suspeita de melanoma:  MEL é a mais provável, ou P(MEL) >= limiar
        MEDIA           suspeita de câncer não melanoma: BCC ou SCC é a mais provável,
                        ou P(BCC) + P(SCC) >= limiar
        ACOMPANHAMENTO  ACK (pré-maligna) é a classe mais provável
        ROTINA          nenhum dos critérios acima
     BCC e SCC ficam no mesmo nível porque o modelo não separa bem os dois
     (nem SCC de ACK), e clinicamente costumam seguir o mesmo encaminhamento.
     Cada limiar é escolhido para atingir uma sensibilidade-alvo.
  3. AVALIAÇÃO HONESTA: calibração e limiares são ajustados em 4/5 dos pacientes e
     aplicados no 1/5 restante (validação cruzada agrupada por paciente).
     Intervalos de confiança: bootstrap reamostrando PACIENTES.

Também mostra a prévia da regra clínica "cresceu OU mudou = no mínimo MEDIA"
('não sei' não altera). Os critérios finais da Etapa 4 devem ser validados
com um profissional de saúde.

IMPORTANTE para o app (Etapa 5): as probabilidades usadas aqui são a MÉDIA dos
classificadores das várias seeds (um conjunto de modelos). Para o app ter o mesmo
desempenho, a inferência deve usar os mesmos N classificadores e tirar a média.

Saídas:
  config_triagem.json       calibrador + limiares finais, para a inferência
  triagem_calibracao.png    diagrama de confiabilidade (antes x depois)
  triagem_niveis.csv        matriz diagnóstico real x nível de prioridade

Exemplos:
  python avaliar_triagem.py
  python avaliar_triagem.py --alvo_mel 0.85 --alvo_nm 0.90
"""

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CANCER = ["BCC", "MEL", "SCC"]
NAO_MELANOMA = ["BCC", "SCC"]
BENIGNAS = ["NEV", "SEK"]
NIVEIS = ["ROTINA", "ACOMPANHAMENTO", "MEDIA", "ALTA"]  # índice = gravidade
ROTINA, ACOMP, MEDIA, ALTA = range(4)

ARQ_CONFIG = "config_triagem.json"
ARQ_GRAFICO = "triagem_calibracao.png"
ARQ_MATRIZ = "triagem_niveis.csv"


# ---------------------------------------------------------------------------
# 1. Calibração: temperatura + viés por classe
# ---------------------------------------------------------------------------
def _log_probs(probs):
    return np.log(np.clip(probs, 1e-7, 1.0))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar_calibrador(probs, y):
    """Ajusta T (temperatura) e b (viés por classe) minimizando a log-verossimilhança."""
    logp = _log_probs(probs)
    n_classes = probs.shape[1]

    def nll(params):
        T = np.exp(params[0])
        b = params[1:]
        z = logp / T + b
        z = z - z.max(axis=1, keepdims=True)
        log_sm = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -log_sm[np.arange(len(y)), y].mean() + 1e-3 * np.sum(b ** 2)

    res = minimize(nll, np.zeros(1 + n_classes), method="L-BFGS-B")
    return float(np.exp(res.x[0])), res.x[1:].astype(float)


def aplicar_calibrador(probs, T, b):
    return _softmax(_log_probs(probs) / T + b)


# ---------------------------------------------------------------------------
# 2. Limiares e níveis de prioridade
# ---------------------------------------------------------------------------
def _scores(pcal, classes):
    """Pontuações usadas pelos níveis: P(MEL) e P(BCC) + P(SCC)."""
    p_mel = pcal[:, classes.index("MEL")]
    p_nm = pcal[:, [classes.index(c) for c in NAO_MELANOMA]].sum(axis=1)
    return p_mel, p_nm


def escolher_limiares(pcal, y, classes, alvos):
    """
    Para cada grupo grave, escolhe o maior limiar que ainda coloca pelo menos
    alvo% dos casos reais do grupo acima dele.
    """
    p_mel, p_nm = _scores(pcal, classes)
    e_mel = y == classes.index("MEL")
    e_nm = np.isin(y, [classes.index(c) for c in NAO_MELANOMA])

    def limiar(score, mask, alvo):
        return float(np.quantile(score[mask], 1 - alvo, method="lower")) if mask.any() else 1.0

    return {
        "MEL": limiar(p_mel, e_mel, alvos["MEL"]),
        "NAO_MELANOMA": limiar(p_nm, e_nm, alvos["NAO_MELANOMA"]),
    }


def atribuir_niveis(pcal, classes, limiares):
    """
    Cada imagem recebe o nível MAIS GRAVE cujo critério ela atinge.
    Piso: a classe mais provável sempre leva, no mínimo, ao seu próprio nível.
    """
    mais_provavel = pcal.argmax(axis=1)
    p_mel, p_nm = _scores(pcal, classes)

    niv = np.full(len(pcal), ROTINA, dtype=int)
    niv[mais_provavel == classes.index("ACK")] = ACOMP
    niv[np.isin(mais_provavel, [classes.index(c) for c in NAO_MELANOMA])] = MEDIA
    niv[mais_provavel == classes.index("MEL")] = ALTA

    niv[p_nm >= limiares["NAO_MELANOMA"]] = np.maximum(niv[p_nm >= limiares["NAO_MELANOMA"]], MEDIA)
    niv[p_mel >= limiares["MEL"]] = ALTA
    return niv


def aplicar_regra_evolucao(niv, sinal):
    """Prévia da Etapa 4: cresceu OU mudou = no mínimo MEDIA. 'Não sei' não altera."""
    out = niv.copy()
    out[sinal] = np.maximum(out[sinal], MEDIA)
    return out


# ---------------------------------------------------------------------------
# 3. Métricas
# ---------------------------------------------------------------------------
def brier(score, alvo):
    return float(np.mean((score - alvo) ** 2))


def ece(score, alvo, n_bins=10):
    bordas = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(score, bordas) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += abs(alvo[m].mean() - score[m].mean()) * m.mean()
    return float(total)


def metricas_niveis(y, niv, classes):
    def frac(mask, cond):
        return float(cond[mask].mean()) if mask.any() else np.nan

    idx = {c: classes.index(c) for c in classes}
    e_cancer = np.isin(y, [idx[c] for c in CANCER])
    e_benigna = np.isin(y, [idx[c] for c in BENIGNAS])
    e_mel, e_scc, e_bcc, e_ack = (y == idx[c] for c in ("MEL", "SCC", "BCC", "ACK"))
    return {
        "Melanomas em ALTA": frac(e_mel, niv == ALTA),
        "Melanomas em ROTINA/ACOMP. (perdidos)": frac(e_mel, niv <= ACOMP),
        "SCC em MEDIA ou acima": frac(e_scc, niv >= MEDIA),
        "BCC em MEDIA ou acima": frac(e_bcc, niv >= MEDIA),
        "Cânceres em MEDIA ou acima": frac(e_cancer, niv >= MEDIA),
        "Cânceres em ROTINA/ACOMP. (perdidos)": frac(e_cancer, niv <= ACOMP),
        "Benignas (NEV+SEK) em MEDIA ou acima": frac(e_benigna, niv >= MEDIA),
        "Benignas (NEV+SEK) em ALTA": frac(e_benigna, niv == ALTA),
        "ACK em MEDIA ou acima": frac(e_ack, niv >= MEDIA),
        "Total de casos em ALTA": float((niv == ALTA).mean()),
        "Total de casos em MEDIA ou acima": float((niv >= MEDIA).mean()),
    }


def bootstrap_pacientes(fn, grupos, n_boot, seed=0):
    """Reamostra PACIENTES (todas as imagens de um paciente juntas) com reposição."""
    rng = np.random.default_rng(seed)
    pacientes = np.unique(grupos)
    idx_por_pac = [np.where(grupos == p)[0] for p in pacientes]
    valores = []
    for _ in range(n_boot):
        escolha = rng.integers(0, len(pacientes), len(pacientes))
        idx = np.concatenate([idx_por_pac[i] for i in escolha])
        valores.append(fn(idx))
    return pd.DataFrame(valores)


def fmt_ic(valor, amostras, pct=True):
    lo, hi = np.nanpercentile(amostras, [2.5, 97.5])
    if pct:
        return f"{valor:6.1%}  (IC95%: {lo:.1%} a {hi:.1%})"
    return f"{valor:6.3f}  (IC95%: {lo:.3f} a {hi:.3f})"


# ---------------------------------------------------------------------------
# Dados auxiliares: cresceu/mudou (para a prévia da regra)
# ---------------------------------------------------------------------------
def carregar_sinal_evolucao(img_ids):
    csv = os.path.join(os.getenv("PAD_PATH", "."), "metadata.csv")
    if not os.path.exists(csv):
        print(f">> AVISO: {csv} não encontrado; a prévia da regra cresceu/mudou será pulada.")
        return None
    meta = pd.read_csv(csv).set_index("img_id")

    def sim(col):
        return meta[col].astype(str).str.strip().str.upper() == "TRUE"

    sinal = (sim("grew") | sim("changed")).reindex(img_ids)
    if sinal.isna().any():
        print(">> AVISO: imagens sem metadados; a prévia da regra cresceu/mudou será pulada.")
        return None
    return sinal.values.astype(bool)


# ---------------------------------------------------------------------------
def main(args):
    dados = np.load(args.oof, allow_pickle=True)
    classes = [str(c) for c in dados["classes"]]
    y = dados["y"].astype(int)
    grupos = dados["patient_id"]
    img_ids = [str(i) for i in dados["img_id"]]
    probs = dados["probs"].mean(axis=0)  # média das seeds: cada valor continua fora da amostra
    n_seeds = dados["probs"].shape[0]

    alvos = {"MEL": args.alvo_mel, "NAO_MELANOMA": args.alvo_nm}
    ic = [classes.index(c) for c in CANCER]
    e_cancer = np.isin(y, ic).astype(float)
    mel = classes.index("MEL")
    e_nm = np.isin(y, [classes.index(c) for c in NAO_MELANOMA])

    print(f"Base: {args.oof} | {len(y)} imagens | {len(np.unique(grupos))} pacientes | "
          f"média de {n_seeds} seeds")
    print("Contagem por classe:", dict(zip(classes, np.bincount(y, minlength=len(classes)).tolist())))
    print(f"Prevalência de câncer (MEL+BCC+SCC) no desenvolvimento: {e_cancer.mean():.1%}")
    print(f"Sensibilidades-alvo: melanoma em ALTA = {alvos['MEL']:.0%} | "
          f"BCC+SCC em MEDIA ou acima = {alvos['NAO_MELANOMA']:.0%}")

    # ---- Validação cruzada externa (por paciente) para calibração + limiares ----
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    folds = []
    pcal_oof = np.zeros_like(probs)
    for tr, te in sgkf.split(probs, y, groups=grupos):
        T, b = ajustar_calibrador(probs[tr], y[tr])
        p_tr, p_te = aplicar_calibrador(probs[tr], T, b), aplicar_calibrador(probs[te], T, b)
        pcal_oof[te] = p_te
        folds.append((tr, te, p_tr, p_te))

    def niveis_cv(alvos_):
        niv = np.zeros(len(y), dtype=int)
        for tr, te, p_tr, p_te in folds:
            lim = escolher_limiares(p_tr, y[tr], classes, alvos_)
            niv[te] = atribuir_niveis(p_te, classes, lim)
        return niv

    # ================= 1. Pontuação de suspeita de câncer =================
    score_bruto = probs[:, ic].sum(axis=1)
    score_cal = pcal_oof[:, ic].sum(axis=1)

    print("\n================ 1. PONTUAÇÃO DE SUSPEITA DE CÂNCER ================")
    print(f"{'':<22}{'Brier (menor=melhor)':>22}{'ECE (menor=melhor)':>20}{'AUC':>8}")
    for nome, s in [("Sem calibrar", score_bruto), ("Calibrada", score_cal)]:
        print(f"{nome:<22}{brier(s, e_cancer):>22.4f}{ece(s, e_cancer):>20.4f}"
              f"{roc_auc_score(e_cancer, s):>8.3f}")

    boot_auc = bootstrap_pacientes(
        lambda idx: {"auc": roc_auc_score(e_cancer[idx], score_cal[idx])}, grupos, args.boot)
    print("\nAUC câncer (calibrada):", fmt_ic(roc_auc_score(e_cancer, score_cal), boot_auc["auc"], pct=False))

    print("\nO que a pontuação calibrada significa (validação cruzada):")
    print(f"{'Pontuação':<12}{'Imagens':>9}{'% câncer real':>16}")
    bordas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
    for lo, hi in zip(bordas[:-1], bordas[1:]):
        m = (score_cal >= lo) & (score_cal < hi)
        if m.any():
            print(f"{f'{lo:.0%}-{min(hi, 1):.0%}':<12}{m.sum():>9}{e_cancer[m].mean():>16.1%}")
    print(f"Atenção: vale para a população do PAD-UFES-20 ({e_cancer.mean():.0%} de câncer). "
          f"Em uma UBS a prevalência é outra e esses números mudam.")

    salvar_grafico_calibracao(score_bruto, score_cal, e_cancer)

    # ================= 2. Níveis de prioridade =================
    niv = niveis_cv(alvos)
    print("\n================ 2. NÍVEIS DE PRIORIDADE (validação cruzada) ================")
    tab = pd.crosstab(
        pd.Categorical([classes[i] for i in y], categories=["MEL", "SCC", "BCC", "ACK", "NEV", "SEK"]),
        pd.Categorical([NIVEIS[i] for i in niv], categories=NIVEIS[::-1]),
        rownames=["Diagnóstico real"], colnames=["Nível"], dropna=False,
    )
    print(tab.to_string())
    tab.to_csv(ARQ_MATRIZ)

    m = metricas_niveis(y, niv, classes)
    boot = bootstrap_pacientes(lambda idx: metricas_niveis(y[idx], niv[idx], classes), grupos, args.boot)
    print(f"\nMétricas de segurança (IC95% por bootstrap de pacientes, {args.boot} repetições):")
    for k, v in m.items():
        print(f"  {k:<40} {fmt_ic(v, boot[k])}")

    # ---- Prévia da regra cresceu/mudou ----
    sinal = carregar_sinal_evolucao(img_ids)
    if sinal is not None:
        niv_r = aplicar_regra_evolucao(niv, sinal)
        m_r = metricas_niveis(y, niv_r, classes)
        print(f"\nPrévia da regra 'cresceu OU mudou = no mínimo MEDIA' "
              f"({sinal.mean():.0%} das imagens têm o sinal):")
        print(f"  {'':<40} {'só modelo':>10} {'+ regra':>10}")
        for k in m:
            print(f"  {k:<40} {m[k]:>10.1%} {m_r[k]:>10.1%}")

    # ================= 3. Escolha dos alvos =================
    print("\n================ 3. ESCOLHA DOS ALVOS ================")
    print("a) Melanoma (alvo de não melanoma fixo):")
    print(f"{'Alvo MEL':>9} | {'MEL em ALTA':>11} | {'MEL perdidos':>12} | {'Não-MEL em ALTA':>15} | {'Total em ALTA':>13}")
    for a in [0.70, 0.80, 0.85, 0.90, 0.95]:
        n_ = niveis_cv({**alvos, "MEL": a})
        print(f"{a:>9.0%} | {(n_[y == mel] == ALTA).mean():>11.1%} | {(n_[y == mel] <= ACOMP).sum():>12d} | "
              f"{((n_ == ALTA) & (y != mel)).sum():>15d} | {(n_ == ALTA).mean():>13.1%}")
    print(f"   (Há {(y == mel).sum()} melanomas: cada um vale {1 / max((y == mel).sum(), 1):.1%} da sensibilidade.)")

    e_benigna = np.isin(y, [classes.index(c) for c in BENIGNAS])
    e_ack = y == classes.index("ACK")
    print("\nb) Câncer não melanoma (alvo de melanoma fixo):")
    print(f"{'Alvo BCC+SCC':>12} | {'BCC+SCC em MEDIA+':>17} | {'BCC+SCC perdidos':>16} | "
          f"{'ACK em MEDIA+':>13} | {'Benignas em MEDIA+':>18} | {'Total MEDIA+':>12}")
    for a in [0.70, 0.80, 0.85, 0.90, 0.95]:
        n_ = niveis_cv({**alvos, "NAO_MELANOMA": a})
        print(f"{a:>12.0%} | {(n_[e_nm] >= MEDIA).mean():>17.1%} | {(n_[e_nm] <= ACOMP).sum():>16d} | "
              f"{(n_[e_ack] >= MEDIA).mean():>13.1%} | {(n_[e_benigna] >= MEDIA).mean():>18.1%} | "
              f"{(n_ >= MEDIA).mean():>12.1%}")

    # ================= 4. Configuração final para o app =================
    T, b = ajustar_calibrador(probs, y)
    p_all = aplicar_calibrador(probs, T, b)
    limiares = escolher_limiares(p_all, y, classes, alvos)
    config = {
        "criado_em": str(date.today()),
        "base": args.oof,
        "n_seeds": int(n_seeds),
        "probabilidades": f"média dos {n_seeds} classificadores (uma seed cada); o app deve fazer o mesmo",
        "classes": classes,
        "classes_cancer": CANCER,
        "calibracao": {"temperatura": T, "vies": dict(zip(classes, b.tolist()))},
        "sensibilidades_alvo": alvos,
        "limiares": limiares,
        "niveis": NIVEIS,
        "regras_niveis": {
            "ALTA": "MEL é a classe mais provável, ou P(MEL) calibrada >= limiares.MEL",
            "MEDIA": "BCC ou SCC é a classe mais provável, ou P(BCC)+P(SCC) calibrada >= limiares.NAO_MELANOMA",
            "ACOMPANHAMENTO": "ACK é a classe mais provável",
            "ROTINA": "nenhum critério acima",
            "combinacao": "vale o nível mais grave atingido",
            "regra_clinica_previa": "cresceu OU mudou = no mínimo MEDIA ('não sei' não altera) — validar na Etapa 4",
        },
    }
    with open(ARQ_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("\n================ 4. CONFIGURAÇÃO FINAL ================")
    print(f"Temperatura: {T:.3f} | limiares: " + ", ".join(f"{k}={v:.3f}" for k, v in limiares.items()))
    print(f">> Salvo em {ARQ_CONFIG} (usado na inferência, Etapa 5)")


def salvar_grafico_calibracao(score_bruto, score_cal, alvo):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(">> matplotlib não instalado; gráfico de calibração não gerado.")
        return

    bordas = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9a9a9a", linewidth=1, label="Calibração perfeita")
    for score, nome, cor in [(score_bruto, "Sem calibrar", "#c2410c"), (score_cal, "Calibrada", "#1d4ed8")]:
        idx = np.clip(np.digitize(score, bordas) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(score[m].mean())
                ys.append(alvo[m].mean())
        ax.plot(xs, ys, marker="o", color=cor, linewidth=2, label=nome)
    ax.set_xlabel("Pontuação de suspeita prevista")
    ax.set_ylabel("Fração real de câncer")
    ax.set_title("Calibração da pontuação de suspeita\n(PAD-UFES-20, validação cruzada)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(ARQ_GRAFICO, dpi=150)
    plt.close(fig)
    print(f">> Gráfico salvo em {ARQ_GRAFICO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etapa 3: suspeita de câncer e níveis de prioridade")
    parser.add_argument("--oof", default="oof_sintomas.npz")
    parser.add_argument("--alvo_mel", type=float, default=0.90,
                        help="fração dos melanomas que deve cair em ALTA")
    parser.add_argument("--alvo_nm", type=float, default=0.85,
                        help="fração dos BCC+SCC que deve cair em MEDIA ou acima")
    parser.add_argument("--boot", type=int, default=1000, help="repetições do bootstrap")
    main(parser.parse_args())
