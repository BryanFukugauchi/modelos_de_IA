from model import build_model
from train import train_pipeline
from infer import predict

class EfficientNetV2:
    def __init__(self):
        self.model = build_model()
    
    def train(self):
        train_pipeline()

    def predict(self, image):
        return predict(image, self.model)