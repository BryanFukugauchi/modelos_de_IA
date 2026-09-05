from train import train
from infer import predict, CLASSES_HAM10000

class ConvNeXt:
    def __init__(self, architecture="tiny"):
        """
        :param architecture: 'tiny' ou 'small'
        """
        self.architecture = architecture
        self.classes = CLASSES_HAM10000

    def train(self, epochs_cabeca=10, epochs_fine_tuning=10, batch_size=32,
              save_path="convnext_model.keras"):
        train(
            epochs_cabeca=epochs_cabeca,
            epochs_fine_tuning=epochs_fine_tuning,
            batch_size=batch_size,
            architecture=self.architecture,
            save_path=save_path
        )

    def predict(self, image_path, model_path="convnext_model.keras"):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        pred_class, confidence = predict(image_path, model)
        return self.classes.get(pred_class, "Desconhecido"), confidence