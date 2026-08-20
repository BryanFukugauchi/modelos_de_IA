from model import build_model
from train import train
from infer import predict, CLASSES_HAM10000

class ConvNeXt:
    def __init__(self, architecture="tiny"):
        """
        :param architecture: 'tiny' ou 'small'
        """
        self.architecture = architecture
        self.classes = CLASSES_HAM10000

    def train(self, epochs=5, batch_size=16, save_path="convnext_model.keras"):
        train(
            epochs=epochs,
            batch_size=batch_size,
            architecture=self.architecture,
            save_path=save_path
        )

    def predict(self, image_path, model_path="convnext_model.keras"):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        pred_class, confidence = predict(image_path, model)
        return self.classes.get(pred_class, "Desconhecido"), confidence