import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# 1. Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# 2. Busca a variável DATASET_PATH do .env. Se não existir, usa a pasta atual ('.')
DATASET_PATH = os.getenv("DATASET_PATH", ".")

def carregar_dataset(caminho_raiz=DATASET_PATH):
    """
    Carrega o arquivo HAM10000_metadata.csv e localiza o caminho completo
    de cada imagem nas pastas part_1, part_2 ou na pasta images unificada.
    """
    csv_path = os.path.join(caminho_raiz, "HAM10000_metadata.csv")
   
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"O arquivo {csv_path} não foi encontrado. "
            f"Verifique se o caminho no seu .env (DATASET_PATH={caminho_raiz}) está correto."
        )
       
    df = pd.read_csv(csv_path)

    # Locais onde as fotos podem estar armazenadas dentro da pasta do dataset
    pasta_part1 = os.path.join(caminho_raiz, "HAM10000_images_part_1")
    pasta_part2 = os.path.join(caminho_raiz, "HAM10000_images_part_2")
    pasta_images = os.path.join(caminho_raiz, "images")

    def resolver_caminho_imagem(image_id):
        nome_arquivo = f"{image_id}.jpg"

        # Tenta achar na Parte 1
        caminho_p1 = os.path.join(pasta_part1, nome_arquivo)
        if os.path.exists(caminho_p1):
            return caminho_p1

        # Tenta achar na Parte 2
        caminho_p2 = os.path.join(pasta_part2, nome_arquivo)
        if os.path.exists(caminho_p2):
            return caminho_p2

        # Tenta achar na pasta comum 'images' (caso tenha unificado)
        caminho_img = os.path.join(pasta_images, nome_arquivo)
        if os.path.exists(caminho_img):
            return caminho_img

        raise FileNotFoundError(f"A imagem {nome_arquivo} não foi localizada em nenhuma das pastas dentro de: {caminho_raiz}")

    print(f"Mapeando o local das imagens a partir de: {caminho_raiz}")
    df["path"] = df["image_id"].apply(resolver_caminho_imagem)
   
    # Mapeia as classes de diagnóstico (dx) em números de 0 a 6
    if "dx" in df.columns:
        df["label"] = pd.Categorical(df["dx"]).codes

    print(f"Sucesso! {len(df)} imagens foram localizadas.")
    return df

def criar_splits(df): 
    """Separa em 80% Treino, 10% Validação e 10% Teste mantendo a proporção das doenças.""" 
    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"]) 
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42, stratify=test_val_df["label"]) 
    return train_df, val_df, test_df

load_ham10000_data = carregar_dataset      

if __name__ == "__main__":
    # Teste rápido ao executar o script diretamente
    try:
        dataframe = carregar_dataset()
        print("\nExemplo das primeiras linhas encontradas:")
        print(dataframe[["image_id", "label", "path"]].head())
    except Exception as e:
        print(f"\n[ERRO]: {e}")
 