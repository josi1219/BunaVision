# Models Directory for BunaVision AI Engine

Place your trained YOLOv8 model weights here:
- **`best.pt`**: Fine-tuned YOLOv8 nano model trained on the green coffee defect dataset using `train_yolov8_coffee.ipynb` in Google Colab.

### How it works:
When `best.pt` is placed in this directory:
1. `app/services/yolo_detector.py` automatically detects and loads it.
2. The API uses real neural network inference to predict bounding box coordinates and defect categories.
3. If `best.pt` is not yet present, the service runs in demonstration/mock mode with graceful fallback so the backend never crashes.
