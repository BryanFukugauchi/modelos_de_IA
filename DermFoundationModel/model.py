"""
model.py — Extratores de embedding do Derm Foundation e o classificador
leve treinado em cima desses embeddings.

IMPORTANTE:
  1. Não está em tfhub.dev — é distribuído via Hugging Face Hub, com
     acesso restrito (é preciso aceitar os termos de uso antes de baixar).
  2. Espera imagens PNG de 448x448 dentro de um tf.train.Example serializado.
  3. Devolve um vetor de embedding de 6144 dimensões.
"""

from __future__ import annotations

from io import BytesIO
import os

from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

EMBEDDING_DIM = 6144
DERM_FOUNDATION_INPUT_SIZE = (448, 448)


def carregar_extrator_de_embeddings():
    """Carrega o modelo REAL do Derm Foundation via Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Pacote 'huggingface_hub' não instalado. Rode: pip install huggingface_hub"
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    try:
        pasta_local = snapshot_download(repo_id="google/derm-foundation", token=token)
        # Carrega na CPU: parte do modelo roda via XlaCallModule (StableHLO/JAX
        # convertido para TF), compilado pelo Google especificamente para a
        # plataforma CPU — confirmado por um NotFoundError explícito ao tentar
        # rodar na GPU. Não é uma escolha de configuração, é uma limitação do
        # próprio artefato do modelo. A inferência em imagem_para_embedding()
        # também precisa ficar consistente na CPU.
        with tf.device("/CPU:0"):
            modelo = tf.saved_model.load(pasta_local)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível carregar o Derm Foundation do Hugging Face. "
            "Confirme que você aceitou os termos de uso em "
            "https://huggingface.co/google/derm-foundation e configurou "
            "a variável de ambiente HF_TOKEN."
        ) from exc

    return modelo.signatures["serving_default"]


def imagem_para_embedding(caminho_imagem: str, infer_fn) -> tf.Tensor:
    """
    Converte uma imagem em disco no formato exato que o Derm Foundation
    espera (PNG 448x448 dentro de um tf.train.Example serializado) e
    devolve o vetor de embedding de 6144 dimensões.
    """
    # 1. Processa e redimensiona a imagem
    img = Image.open(caminho_imagem).convert("RGB").resize(DERM_FOUNDATION_INPUT_SIZE)
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    # 2. Serializa no formato tf.train.Example
    exemplo = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[buffer.getvalue()])
                )
            }
        )
    ).SerializeToString()

    # Confirmado por teste real: parte do modelo (XlaCallModule) só roda em
    # CPU — precisa ficar no mesmo dispositivo de onde foi carregado.
    with tf.device("/CPU:0"):
        saida = infer_fn(inputs=tf.constant([exemplo]))

    return tf.reshape(saida["embedding"], [-1])


def carregar_extrator_substituto() -> tf.keras.Model:
    """
    Extrator alternativo, local e sem necessidade de autenticação:
    ConvNeXtTiny pré-treinado na ImageNet, congelado, usado só para gerar
    um vetor de características por imagem.
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


def build_model(num_classes: int = 7, embedding_dim: int = EMBEDDING_DIM) -> tf.keras.Model:
    """
    Classificador leve que recebe os embeddings e aprende a mapeá-los para as classes.
    """
    entradas = tf.keras.Input(shape=(embedding_dim,))
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(entradas)
    x = layers.Dropout(0.5)(x)
    saidas = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(entradas, saidas, name="DermFoundation_Classifier")