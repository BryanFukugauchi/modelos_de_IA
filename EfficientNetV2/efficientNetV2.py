import tensorflow as tf
from train import train
from infer import predict, CLASSES_HAM10000

class EfficientNetV2:
    def __init__(self, save_path="ham10000_efficientnetv2.keras"):
        self.save_path = save_path
        self.classes = CLASSES_HAM10000
        self.model = None

    def train(self, epochs=5, batch_size=16):
        # Atualiza a referência interna self.model com o modelo recém-treinado
        self.model = train(epochs=epochs, batch_size=batch_size, save_path=self.save_path)

    def predict(self, image_path):
        # Se não houver modelo em memória, carrega o arquivo do disco
        if self.model is None:
            self.model = tf.keras.models.load_model(self.save_path, compile=False)

        pred_class, confidence = predict(image_path, self.model)
        return self.classes.get(pred_class, "Desconhecido"), confidence