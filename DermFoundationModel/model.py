"""
model.py — Extratores de embedding do Derm Foundation e o classificador
leve treinado em cima desses embeddings.

IMPORTANTE (motivo da reescrita): o Derm Foundation NÃO é um backbone de
imagem "normal" como o EfficientNetV2/ConvNeXt:

  1. Não está em tfhub.dev — é distribuído via Hugging Face Hub, com
     acesso restrito (é preciso aceitar os termos de uso do Google
     Health AI Developer Foundations antes de baixar os pesos).
  2. Espera imagens PNG de 448x448 dentro de um tf.train.Example
     serializado — não aceita um tensor de pixels puro.
  3. Só devolve um vetor de embedding de 6144 dimensões — não é uma rede
     de classificação fim-a-fim. O uso oficial recomendado é: gerar os
     embeddings uma vez (backbone congelado) e treinar um classificador
     pequeno em cima deles.

Por isso este módulo expõe duas partes separadas: o extrator de
embeddings (real ou substituto) e o classificador leve, que é o único
componente de fato treinado.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

EMBEDDING_DIM = 6144
DERM_FOUNDATION_INPUT_SIZE = (448, 448)


def carregar_extrator_de_embeddings():
    """
    Carrega o modelo REAL do Derm Foundation via Hugging Face Hub.

    Pré-requisitos (feitos uma única vez, fora do código):
      1. Ter uma conta no Hugging Face.
      2. Acessar https://huggingface.co/google/derm-foundation e aceitar
         os termos de uso.
      3. Gerar um token em https://huggingface.co/settings/tokens e
         exportar: export HUGGINGFACE_HUB_TOKEN=seu_token_aqui

    Nota técnica: a função huggingface_hub.from_pretrained_keras foi
    REMOVIDA na huggingface_hub v1.0 (integração com Keras 2 descontinuada).
    Por isso baixamos o repositório manualmente com snapshot_download() e
    carregamos com tf.keras.models.load_model() — é exatamente o que
    from_pretrained_keras fazia internamente, só que sem depender de uma
    função que pode sumir de novo em versões futuras.

    Levanta RuntimeError em vez de mascarar o problema — assim nunca se
    treina "Derm Foundation" sem saber que, na verdade, caiu para outro
    backbone.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Pacote 'huggingface_hub' não instalado. Rode: pip install huggingface_hub"
        ) from exc

    try:
        pasta_local = snapshot_download(repo_id="google/derm-foundation")
        modelo = tf.keras.models.load_model(pasta_local)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível carregar o Derm Foundation do Hugging Face. "
            "Confirme que você aceitou os termos de uso em "
            "https://huggingface.co/google/derm-foundation e configurou "
            "a variável de ambiente HUGGINGFACE_HUB_TOKEN."
        ) from exc

    return modelo.signatures["serving_default"]


def imagem_para_embedding(caminho_imagem: str, infer_fn) -> tf.Tensor:
    """
    Converte uma imagem em disco no formato exato que o Derm Foundation
    espera (PNG 448x448 dentro de um tf.train.Example serializado) e
    devolve o vetor de embedding de 6144 dimensões.
    """
    from io import BytesIO

    from PIL import Image

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

    saida = infer_fn(inputs=tf.constant([exemplo]))
    return tf.reshape(saida["embedding"], [-1])


def carregar_extrator_substituto() -> tf.keras.Model:
    """
    Extrator alternativo, local e sem necessidade de autenticação:
    ConvNeXtTiny pré-treinado na ImageNet, congelado, usado só para gerar
    um vetor de características por imagem.

    ATENÇÃO: isto NÃO é o Derm Foundation real — use apenas enquanto não
    tiver configurado o acesso via Hugging Face (Opção B).
    """
    base = tf.keras.applications.ConvNeXtTiny(
        include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    return base


def imagem_para_embedding_substituto(caminho_imagem: str, extrator: tf.keras.Model) -> tf.Tensor:
    """Gera um vetor de características usando o extrator substituto."""
    img = tf.keras.utils.load_img(caminho_imagem, target_size=(224, 224))
    array = tf.keras.utils.img_to_array(img)
    array = tf.expand_dims(array, axis=0)
    return tf.reshape(extrator(array, training=False), [-1])


def build_model(num_classes: int = 7) -> tf.keras.Model:
    """
    O componente treinável de fato: um classificador leve que recebe
    embeddings JÁ CALCULADOS e aprende a mapeá-los para as classes do
    HAM10000. O backbone pesado fica de fora deste grafo — é congelado e
    usado só como pré-processamento, conforme a documentação oficial.

    Regularização mais forte que a versão anterior (L2 + dropout maior):
    como o classificador é a única parte treinável, com poucas dezenas
    de imagens por classe ele overfita rápido nos embeddings de treino
    se não for contido.
    """
    entradas = tf.keras.Input(shape=(EMBEDDING_DIM,))
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(entradas)
    x = layers.Dropout(0.5)(x)
    saidas = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(entradas, saidas, name="DermFoundation_Classifier")