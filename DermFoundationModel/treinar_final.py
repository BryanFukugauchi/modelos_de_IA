"""
treinar_final.py — Etapa 5: treina os classificadores FINAIS e monta o pacote do modelo.

Configuração escolhida nas Etapas 2 e 3:
  - metadados 'sintomas': foto (embedding Derm Foundation) + idade + região
    + sangramento, coceira, dor, elevação
  - 5 classificadores (seeds 42 a 46), cuja MÉDIA é usada na triagem
    (as probabilidades do avaliar_triagem.py vieram dessa mesma média)

Usa TODOS os pacientes de desenvolvimento. O conjunto reservado (holdout)
continua de fora e só será usado na avaliação final.

Cada classificador separa 20% dos pacientes (um sorteio diferente por seed)
só para o EarlyStopping saber quando parar, como na validação cruzada.

Saída: a pasta modelos_finais/, que é o PACOTE que a API carrega:
  modelos_finais/
    classificador_seed42.keras ... classificador_seed46.keras
    esquema.json          classes, ordem das features, codificação do formulário
    config_triagem.json   cópia da calibração e dos limiares (avaliar_triagem.py)

Rode depois do avaliar_triagem.py --alvo_nm 0.95:
  python treinar_final.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model, EMBEDDING_DIM, DERM_FOUNDATION_INPUT_SIZE  # noqa: E402
from pad_ufes import (  # noqa: E402
    ARQUIVO_HOLDOUT,
    REGIOES_PAD,
    SINTOMAS,
    calcular_ou_carregar_embeddings,
    carregar_pad,
    montar_X,
    remover_holdout,
)

NIVEL_META = "sintomas"
PASTA = "modelos_finais"


def nomes_das_features():
    """Mesma ordem de meta_proprio(df, nivel='sintomas') no pad_ufes.py."""
    return (
        [f"embedding_{i}" for i in range(EMBEDDING_DIM)]
        + ["age"]
        + [f"reg_{r}" for r in REGIOES_PAD]
        + ["reg_OUTRA"]
        + [f"{c}_sim" for c in SINTOMAS]
    )


def treinar_um(X, y, grupos, n_classes, seed, epochs):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    treino, val = next(sgkf.split(X, y, groups=grupos))

    cls = np.unique(y[treino])
    pesos = np.sqrt(compute_class_weight("balanced", classes=cls, y=y[treino]))
    class_weight = dict(zip(cls.tolist(), pesos.tolist()))

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    modelo = build_model(num_classes=n_classes, embedding_dim=X.shape[1], X_adapt=X[treino])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = modelo.fit(
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
    melhor = int(np.argmin(hist.history["val_loss"]))
    return modelo, {
        "seed": seed,
        "imagens_treino": int(len(treino)),
        "imagens_val": int(len(val)),
        "melhor_epoca": melhor + 1,
        "val_loss": float(hist.history["val_loss"][melhor]),
        "val_acuracia": float(hist.history["val_accuracy"][melhor]),
    }


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main(args):
    if not os.path.exists("config_triagem.json"):
        raise SystemExit("config_triagem.json não encontrado. Rode antes: python avaliar_triagem.py --alvo_nm 0.95")
    with open("config_triagem.json", encoding="utf-8") as f:
        config = json.load(f)

    df = carregar_pad()
    emb = calcular_ou_carregar_embeddings(df)
    df, emb = remover_holdout(df, emb)

    classes = sorted(df["diagnostic"].unique())
    if classes != config["classes"]:
        raise SystemExit(f"Classes diferentes do config_triagem.json: {classes} x {config['classes']}")
    y = df["diagnostic"].map({c: i for i, c in enumerate(classes)}).values.astype(int)
    grupos = df["patient_id"].values

    X = montar_X(df, emb, NIVEL_META)
    nomes = nomes_das_features()
    if X.shape[1] != len(nomes):
        raise SystemExit(f"X tem {X.shape[1]} colunas, mas o esquema lista {len(nomes)}.")
    print(f"\nEntradas por imagem: {X.shape[1]} | imagens: {len(y)} | pacientes: {len(np.unique(grupos))}")

    seeds = [42 + i for i in range(args.seeds)]
    if args.seeds != config.get("n_seeds"):
        print(f">> AVISO: a calibração usou {config.get('n_seeds')} seeds e aqui serão {args.seeds}.")

    if os.path.exists(PASTA):
        shutil.rmtree(PASTA)
    os.makedirs(PASTA)

    resumo = []
    for s in seeds:
        modelo, info = treinar_um(X, y, grupos, len(classes), s, args.epochs)
        arquivo = f"classificador_seed{s}.keras"
        modelo.save(os.path.join(PASTA, arquivo))
        info["arquivo"] = arquivo
        resumo.append(info)
        print(f"  seed {s}: melhor época {info['melhor_epoca']:>2} | "
              f"val_loss {info['val_loss']:.3f} | val_acurácia {info['val_acuracia']:.3f} -> {arquivo}")

    # Conferência: recarrega os modelos salvos e checa que a média funciona
    recarregados = [tf.keras.models.load_model(os.path.join(PASTA, r["arquivo"]), compile=False)
                    for r in resumo]
    amostra = X[:8]
    media = np.mean([m.predict(amostra, verbose=0) for m in recarregados], axis=0)
    assert media.shape == (len(amostra), len(classes))
    assert np.allclose(media.sum(axis=1), 1.0, atol=1e-4)
    print(">> Conferência: os modelos salvos recarregam e a média das probabilidades soma 1.")

    esquema = {
        "criado_em": str(date.today()),
        "classes": classes,
        "nivel_metadados": NIVEL_META,
        "n_entradas": int(X.shape[1]),
        "embedding": {
            "modelo": "google/derm-foundation (Hugging Face)",
            "dimensao": EMBEDDING_DIM,
            "preprocessamento": (
                f"RGB, redimensionada para {DERM_FOUNDATION_INPUT_SIZE[0]}x{DERM_FOUNDATION_INPUT_SIZE[1]} "
                "(PIL resize), salva como PNG dentro de um tf.train.Example "
                "com a chave 'image/encoded'; inferência na CPU"
            ),
        },
        "formulario": {
            "idade": "anos; entra como idade/100; ausente = 0",
            "regiao": {"valores": REGIOES_PAD, "outra_regiao": "reg_OUTRA = 1"},
            "sintomas": {
                "campos": SINTOMAS,
                "codificacao": "sim = 1; não ou não sei = 0",
            },
            "fora_do_modelo": ["grew", "changed", "gender", "skin_cancer_history", "cancer_history"],
            "alerta_clinico": "grew OU changed = sim -> alerta para o profissional (não altera o nível)",
        },
        "ordem_das_features": nomes[EMBEDDING_DIM:],
        "nota_ordem": "as primeiras 6144 entradas são o embedding; depois vêm as features acima, nessa ordem",
        "classificadores": resumo,
        "agregacao": "média simples das probabilidades dos classificadores",
        "holdout_excluido": {
            "arquivo": ARQUIVO_HOLDOUT,
            "sha256": sha256(ARQUIVO_HOLDOUT),
        },
    }
    with open(os.path.join(PASTA, "esquema.json"), "w", encoding="utf-8") as f:
        json.dump(esquema, f, ensure_ascii=False, indent=2)
    shutil.copy2("config_triagem.json", os.path.join(PASTA, "config_triagem.json"))

    print(f"\n>> Pacote salvo em {PASTA}/: {len(resumo)} classificadores + esquema.json + config_triagem.json")
    print(">> É essa pasta que a API vai carregar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina os classificadores finais e monta o pacote do modelo")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    main(parser.parse_args())
