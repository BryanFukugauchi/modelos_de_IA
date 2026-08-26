import tensorflow as tf
import tensorflow_hub as hub
from train import train
from infer import predict, CLASSES_HAM10000

class DermFoundation:
    def __init__(self, save_path="derm_foundation_model.keras"):
        self.save_path = save_path
        self.classes = CLASSES_HAM10000
        self.model = None

    def train(self, epochs=5, batch_size=16):
        # Atualiza a referência em memória
        self.model = train(
            epochs=epochs,
            batch_size=batch_size,
            save_path=self.save_path
        )

    def predict(self, image_path):
        # Se não houver modelo na memória, carrega do disco reconhecendo o KerasLayer
        if self.model is None:
            self.model = tf.keras.models.load_model(
                self.save_path, 
                custom_objects={'KerasLayer': hub.KerasLayer},
                compile=False
            )

        pred_class, confidence = predict(image_path, self.model)
        return self.classes.get(pred_class, "Desconhecido"), confidence