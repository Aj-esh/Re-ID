# YOLOv11 Player Tracking - Project Report

---

## 1. Approach and Methodology

- **Detection**: Utilized Ultralytics' YOLOv11 model to detect players in each frame.
- **Tracking**: Created a custom player tracker with IoU-based matching for frame-to-frame consistency.
- **Re-identification**: When a player exits and re-enters, we reassign their identity using a combination of HSV color histogram signatures and DeepSORT based mobilenet-based embeddings. This hybrid approach enhances re-identification robustness by leveraging both appearance-based color distribution and deep feature embeddings for improved matching accuracy.

---

## 2. Techniques Tried and Outcomes

- **YOLOv11**: Provided strong and consistent detections for the player class.
- **IoU-based tracking**: Works well for players consistently in view.
- **Histogram ReID**: Effective for re-identifying players based on visual features even after a short disappearance.
- **Histogram ReID**: Uses HSV color histograms to capture the color distribution of detected player regions for visual comparison.
- **Embedding ReID**: Utilizes MobileNet-based DeepSORT embeddings to extract deep visual features for robust identity matching even under occlusion or changes in appearance.

---

## 3. Challenges Encountered

- Re-identification sometimes fails when lighting changes drastically.
- False positives from the model affect ID consistency.
- Final seconds of the video is not consistent with the rest, affected re-ID consistency.  
- Using `cv2.imshow()` doesn't work well in colab environments.

---

## 4. Remaining Work and Future Improvements

- Add velocity-based matching for better motion tracking.
- Train a lightweight ReID model for improved player re-identification.
- Save output video with tracking overlays.

