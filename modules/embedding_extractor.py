from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

class EmbeddingExtractor:

    def __init__(self, model_name='resnet50'):
        self.model = self._load_model(model_name)

    def _load_model(self, model_name):
        if model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet')
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        return Model(inputs=base_model.input, outputs=base_model.output)

    def extract_embeddings(self, images):
        preprocessed_images = preprocess_input(images)
        embeddings = self.model.predict(preprocessed_images)
        return embeddings