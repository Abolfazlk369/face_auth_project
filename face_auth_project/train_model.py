import cv2
import os
import numpy as np

def prepare_training_data(data_folder_path):
    faces = []
    labels = []

    for folder_name in os.listdir(data_folder_path):
        if not os.path.isdir(os.path.join(data_folder_path, folder_name)):
            continue

        label = int(folder_name.split("_")[0])
        folder_path = os.path.join(data_folder_path, folder_name)

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"[!] خواندن تصویر شکست خورد: {image_path}")
                continue

            faces.append(image)
            labels.append(label)

    return faces, labels

def train_model():
    print("🧠 در حال آموزش مدل تشخیص چهره...")

    data_path = "data"
    faces, labels = prepare_training_data(data_path)

    if len(faces) == 0:
        print("[×] هیچ چهره‌ای برای آموزش پیدا نشد!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save("trained_model.yml")

    print("[✔] آموزش مدل با موفقیت انجام شد و ذخیره شد به عنوان 'trained_model.yml'.")

if __name__ == "__main__":
    train_model()
