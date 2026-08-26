import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import argparse
import pandas as pd
import tensorflow as tf
from model import build_model

def load_ham10000_data():
    csv_matches = glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)
    if not csv_matches:
        raise FileNotFoundError("HAM10000_metadata.csv não encontrado no diretório /kaggle/input/")

    metadata_path = csv_matches[0]
    base_dir = os.path.dirname(metadata_path)
    print(f"Dataset localizado em: {base_dir}")

    df = pd.read_csv(metadata_path)
    all_image_paths = glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
    image_path_map = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}

    df["path"] = df["image_id"].map(image_path_map)
    df["label"] = pd.Categorical(df["dx"]).codes
    return df

def create_tf_dataset(df, batch_size=16, img_size=(224, 224)):
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, img_size)
        return image, label

    filenames = df["path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train(epochs=5, batch_size=16, save_path="derm_foundation_model.keras"):
    print("Mapeando imagens do HAM10000...")
    df = load_ham10000_data()

    val_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(val_df.index)

    train_ds = create_tf_dataset(train_df, batch_size=batch_size)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size)

    print("Construindo modelo Derm Foundation...")
    model = build_model(input_shape=(224, 224, 3), num_classes=7)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Iniciando treinamento...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save(save_path)
    print(f"Modelo salvo em: {save_path}")

    return model  # RETORNO ADICIONADO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=str, default="derm_foundation_model.keras")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)