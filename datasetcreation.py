import os
import base64
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import logging
import shutil

logging.basicConfig(level=logging.INFO)

class DatasetCreation:
    BASE_DIR = "dataset"
    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

    def __init__(self):
        for cls in self.CLASSES:
            os.makedirs(os.path.join(self.BASE_DIR, cls), exist_ok=True)

        logging.info("Dataset directories created.")

        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["braintumor"]
        self.collection = self.db["analysis"]


    def remove_dataset(self):
        if os.path.exists(self.BASE_DIR):
            shutil.rmtree(self.BASE_DIR)
            logging.info(f"Removed dataset directory: {self.BASE_DIR}")
        else:
            logging.warning("Dataset directory does not exist.")

    @classmethod
    def save_image(cls, base64_img, prediction, index):
        if prediction not in cls.CLASSES:
            logging.warning(f"Invalid class: {prediction}")
            return

        # Remove base64 header if present
        if "," in base64_img:
            base64_img = base64_img.split(",")[1]

        try:
            image_bytes = base64.b64decode(base64_img)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            folder_path = os.path.join(cls.BASE_DIR, prediction)
            file_path = os.path.join(folder_path, f"image_{index}.png")

            image.save(file_path)
            logging.info(f"Saved → {file_path}")

        except Exception as e:
            logging.error(f"Failed to save image {index}: {e}")

    def create_dataset(self):
        for i, doc in enumerate(self.collection.find()):
            prediction = doc.get("prediction")
            base64_img = doc.get("image_base64")

            if prediction and base64_img:
                self.save_image(base64_img, prediction, i)

        logging.info("Dataset creation completed.")
        return os.path.abspath(self.BASE_DIR)

if __name__ == "__main__":
    dataset_creator = DatasetCreation()
    s=dataset_creator.create_dataset()
    print(f"Dataset created at: {s}")
    dataset_creator.remove_dataset()
