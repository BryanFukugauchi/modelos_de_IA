import os
import pandas as pd

def carregar_dataset(caminho_raiz="."):
    """
    Carrega o arquivo HAM10000_metadata.csv e encontra o caminho exato
    de cada imagem nas pastas part_1 ou part_2.
    """
    csv_path = os.path.join(caminho_raiz, "HAM10000_metadata.csv")
   
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"O arquivo {csv_path} não foi encontrado na pasta do projeto.")
       
    df = pd.read_csv(csv_path)

    # Definir os locais possíveis onde as fotos podem estar
    pasta_part1 = os.path.join(caminho_raiz, "HAM10000_images_part_1")
    pasta_part2 = os.path.join(caminho_raiz, "HAM10000_images_part_2")
    pasta_images = os.path.join(caminho_raiz, "images")  # Caso decida mover no futuro

    def resolver_caminho_imagem(image_id):
        nome_arquivo = f"{image_id}.jpg"

        # 1. Tenta achar na Parte 1
        caminho_p1 = os.path.join(pasta_part1, nome_arquivo)
        if os.path.exists(caminho_p1):
            return caminho_p1

        # 2. Tenta achar na Parte 2
        caminho_p2 = os.path.join(pasta_part2, nome_arquivo)
        if os.path.exists(caminho_p2):
            return caminho_p2

        # 3. Tenta achar na pasta images comum (se existir)
        caminho_img = os.path.join(pasta_images, nome_arquivo)
        if os.path.exists(caminho_img):
            return caminho_img

        raise FileNotFoundError(f"A imagem {nome_arquivo} não foi encontrada nas pastas de dados.")

    # Cria uma nova coluna com o caminho final e absoluto de cada foto
    print("Mapeando o local de todas as 10.015 imagens...")
    df["path"] = df["image_id"].apply(resolver_caminho_imagem)
   
    # Criar mapeamento de rótulos (labels) para inteiros se necessário
    if "dx" in df.columns:
        df["label"] = pd.Categorical(df["dx"]).codes

    print(f"Sucesso! {len(df)} imagens foram localizadas.")
    return df

if __name__ == "__main__":
    # Teste rápido ao executar o arquivo diretamente
    dataframe = carregar_dataset()
    print(dataframe[["image_id", "path"]].head())