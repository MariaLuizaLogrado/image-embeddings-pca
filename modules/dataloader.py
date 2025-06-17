import tensorflow as tf
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, path, batch_size=32, shuffle=True, image_size=(224, 224)):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size

    def load_data(self):
        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            labels = 'inferred',
            label_mode='int',
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=self.shuffle
        )

        return self.dataset

    def plot_dataset(self, batches=1):
       for batch, labels in self.dataset.take(batches):
            for i in range(9):  # mostra as 9 primeiras imagens do primeiro batch
                plt.subplot(3, 3, i + 1)
                plt.title(f"Label: {labels[i].numpy()}")
                plt.imshow(batch[i].numpy().astype("uint8"))
                plt.axis("off")
            plt.tight_layout()
            plt.show()
