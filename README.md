# YOLOv11-Based Player Tracking System

This project uses YOLOv11 for object detection and a custom DeepSORT-inspired method for player tracking across video frames, with robust re-identification.

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- Ultralytics
- OpenCV
- NumPy
- torch 
- deep_sort_realtime

Install dependencies manually:

```bash
pip install ultralytics opencv-python numpy torch
```

### Setup

1. Clone this repository or download the code.
2. Place your YOLOv11 `.pt` model file in the working directory.
3. Run the tracking pipeline on your video:

```python
from ultralytics import YOLO
from player_tracker import PlayerTracker
from your_script import process_video

model = YOLO("best.pt")
tracker = PlayerTracker()
process_video("your_video.mp4", model, tracker)
```

---

## Project Structure

```
.
├── player_tracker.py       # Contains the PlayerTracker class
├── process_video.py        # Processes video frame-by-frame
├── yolo11_ultralytics.ipynb # Development notebook
├── README.md
└── report.md               # Project report
```

---

## Output

Each video frame is processed in real-time, and detected players are visually annotated with bounding boxes and assigned unique tracking IDs using OpenCV. These frames are then displayed sequentially, providing a live view of the tracking performance throughout the video playback.

---

## ⚒️ Notes

- If using in Jupyter or Colab, replace `cv2.imshow()` with `matplotlib` for inline rendering.
- The tracker stores "inactive" players and re-identifies them using HSV histogram matching, which compares the color distributions of detected player regions. Incorporating DeepSORT embeddings using a CNN-based feature extractor. These embeddings capture fine-grained visual features like texture, clothing patterns, and color signatures in a high-dimensional space. This embedding-based approach significantly enhances identity re-association, especially under occlusion, changing poses, or light, providing more accurate and stable tracking compared to color histogram methods alone.

