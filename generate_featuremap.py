import cv2
import numpy as np
import os
import time
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()

    save_path = "feature_map_test"

    target_path = r"dataset-2"
    excel = "path_class.xlsx"

    import pandas as pd

    # Membaca file Excel
    df = pd.read_excel(excel)

    # Mengambil kolom 'path' dan 'class'
    paths = df['path'].tolist()
    classes = df['class'].tolist()

    # Menampilkan hasil
    for path, cls in zip(paths, classes):
        try:
            path = os.path.join(target_path, path+'.jpg')
            image = Image.open(path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            heatmap_save_path = os.path.join(save_path, cls, os.path.basename(path))
            r_heatmap = yolo.feature_extraction(image, heatmap_save_path)
            print("success")