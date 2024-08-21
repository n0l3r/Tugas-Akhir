import cv2
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import time
from PIL import Image
from yolo import YOLO
from mlp import MLP

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    classes = ['dingin', 'sedang', 'sejuk']
    yolo = YOLO()
    mlp = MLP()
    mlp.load_state_dict(torch.load('weights/best_mlp.pth'))
    mlp.eval()



    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    args = parser.parse_args()

    path_img = args.img

    
    try:
        path = os.path.join(path_img)
        image = Image.open(path)
    except:
        print('Open Error! Try again!')
    else:
        save_path = "feature_map_output"
        heatmap_save_path = os.path.join(save_path, os.path.basename(path))
        start_time = time.time()
        extract_time = time.time()
        print("Detect & Extracting feature map...")
        feature_map = yolo.feature_extraction(image, heatmap_save_path)
        print(f"Feature map extracted - {time.time() - extract_time:.4f} detik")
        feature = Image.open(heatmap_save_path)
        feature_map = transform(feature).unsqueeze(0)
        
        with torch.no_grad():
            print("Predicting...")
            predict_time = time.time()
            predict = mlp(feature_map)
            _, predicted = torch.max(predict.data, 1)
            print(f"Predicted - {time.time() - predict_time:.4f} detik")
            predicted_label = classes[predicted.item()]
            end_time = time.time()

            print(f'Predict : {predicted_label}')
            print(f'Total Execution Time: {end_time - start_time:.4f} detik')
        



       