from model import build_model
from train import train
from infer import predict

class EfficientNetV2:
    def __init__(self):
        self.model = build_model()
    
    def train(self):
        train()

    def predict(self, image):
        return predict(image, self.model)