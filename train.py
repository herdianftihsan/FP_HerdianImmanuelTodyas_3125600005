"""
Baseline training script - PENSHIP Recruitment Final Project

Flow:
  1. Download the dataset from the participant's Roboflow project (YOLOv8 format).
     If dataset/ already contains data.yaml, the download is skipped.
  2. Train YOLOv8.
  3. Evaluate the best model (best.pt) on the validation split.

Setup:
  1. Fill in ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION in the
     CONFIGURATION section. All three appear in the dataset URL:
         app.roboflow.com/<workspace>/<project>/<version>
     or in the snippet under Download Dataset -> YOLOv8 format -> Show download code.
  2. Set the Roboflow API key as an environment variable. Do NOT write the API
     key in this file, because this file is part of the submission.

Run (replace API_KEY_ROBOFLOW with your Roboflow API key):
  Windows (PowerShell):
      pip install -r requirements.txt
      $env:ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
      python train.py

  Linux / WSL / macOS:
      pip install -r requirements.txt
      export ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
      python train.py

For training on Google Colab, use train_colab.ipynb.

This code is intentionally minimal and MUST be extended by each participant.
See the README and the guidebook (Pengembangan Kode section) for ideas.
"""

import os
import argparse
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # PENGEMBANGAN 1: Menambahkan argumen CLI (argparse) agar gampang eksperimen
    parser = argparse.ArgumentParser(description="Training YOLOv8 untuk PENSHIP")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pilih model (yolov8n.pt / yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epoch")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch")
    args = parser.parse_args()

    # 1. Download Dataset dari Roboflow (Pakai Environment Variable agar aman)
    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    project = rf.workspace("herdians-workspace-f6bfc").project("penship-detection-wbcjs")
    version = project.version(2) # Ganti angka 1 dengan nomor version dataset Anda
    dataset = version.download("yolov8")

    # 2. Load Model YOLO
    model = YOLO(args.model)

    print(f"Mulai training menggunakan model {args.model} selama {args.epochs} epoch...")

    # PENGEMBANGAN 2 & 3: Hyperparameter Tuning (AdamW) dan Augmentasi Data (Cahaya & Rotasi)
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        optimizer="AdamW",     # Tuning optimizer agar lebih stabil
        lr0=0.001,             # Learning rate
        patience=20,           # Early stopping (berhenti jika model mulai overfitting)
        hsv_h=0.015,           # Augmentasi warna (Tahan variasi silau/gelap)
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,          # Augmentasi rotasi (Tahan objek yang difoto miring)
        mosaic=1.0             # Menggabungkan 4 foto jadi 1 agar AI lebih teliti
    )

    # PENGEMBANGAN 4: Auto-export model ke format ONNX untuk deployment di SBC Kapal
    model.export(format="onnx")
    print("Training Selesai! Model terbaik tersimpan di folder runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()