import os
import glob
import argparse
import pandas as pd
import tensorflow as tf
from model import build_model

DATASET_PATH = "/kaggle/input/skin-cancer-mnist-ham10000"

def load_ham10000_data():
    """Localiza o arquivo de metadados automaticamente e mapeia os caminhos das imagens."""
    
    # Busca o arquivo CSV em qualquer subpasta dentro de /kaggle/input/
    csv_matches = glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)
    
    if not csv_matches:
        conteudo_input = os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else []
        raise FileNotFoundError(
            f"\n\n[ERRO] O arquivo HAM10000_metadata.csv não foi encontrado.\n"
            f"Pastas encontradas em /kaggle/input: {conteudo_input}\n"
            f"Verifique se o dataset foi adicionado ao notebook no menu lateral do Kaggle."
        )

    metadata_path = csv_matches[0]
    # Pega o diretório raiz onde o CSV e as imagens estão localizados
    base_dir = os.path.dirname(metadata_path)
    print(f"Dataset localizado com sucesso em: {base_dir}")

    df = pd.read_csv(metadata_path)

    # Mapeia todas as imagens .jpg presentes no diretório raiz do dataset
    all_image_paths = glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
    image_path_map = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths
    }

    df["path"] = df["image_id"].map(image_path_map)
    df["label"] = pd.Categorical(df["dx"]).codes

    return df

def create_tf_dataset(df, batch_size=32, img_size=(224, 224)):
    """Carrega as imagens do disco e redimensiona para 224x224 em tempo de execução."""
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, img_size)
        return image, label

    filenames = df["path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train(epochs=5, batch_size=16, save_path="ham10000_efficientnetv2.keras"):
    print("Mapeando imagens do dataset HAM10000...")
    df = load_ham10000_data()
    print(f"Total de imagens encontradas: {len(df)}")

    # Divisão simples de dados (80% treino, 20% validação)
    val_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(val_df.index)

    train_ds = create_tf_dataset(train_df, batch_size=batch_size)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size)

    # Constrói o modelo com 7 classes
    model = build_model(input_shape=(224, 224, 3), num_classes=7, pretrained=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Iniciando treinamento com HAM10000...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save(save_path)
    print(f"Modelo salvo com sucesso em: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)