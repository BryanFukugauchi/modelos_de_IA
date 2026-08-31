from train import train
from infer import predict, CLASSES_HAM10000

class DermFoundation:
    def __init__(self):
        self.classes = CLASSES_HAM10000

    def train(self, epochs=20, batch_size=32, save_path="derm_foundation_model.keras"):
        train(
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_path
        )

    def predict(self, image_path, model_path="derm_foundation_model.keras"):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        pred_class, confidence = predict(image_path, model)
        return self.classes.get(pred_class, "Desconhecido"), confidence