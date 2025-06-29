import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self, iou=0.7, sign_threshold=0.7):
        self.active_players = {}  # {pid: box}
        self.inactive_players = {}  # {pid: {"box": box, "signature": hist}}

        self.best_match_iou = iou
        self.signature_threshold = sign_threshold
        self.id = 0
        self.frame_count = 0 # Initialize frame_count

        self.deepsort = DeepSort(max_age=100, embedder="mobilenet", half=True)

    def IOU(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        inter = w * h
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

    def signature(self, frame_count, bx, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        x1, y1, x2, y2 = map(int, bx)
        w, h = x2 - x1, y2 - y1
        bbox_xywh = [[x1, y1, w, h]]

        patch = hsv[y1:y2, x1:x2]
        hist = cv2.calcHist([patch], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten()

        embedding = self.deepsort.generate_embeds(img, [bbox_xywh])

        return [hist, embedding]


    def sign_match(self, sign1, sign2):
        c1 = cv2.compareHist(sign1[0].astype('float32'), sign2[0].astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
        c2 = np.linalg.norm(np.array(sign1[1]) - np.array(sign2[1]))

        return (0.23 * c1 + 0.77 * c2)
    
    def update(self, detections, frame_count, img):
        self.frame_count += 1
        matched = {}
        used = []

        # Match active players with current detections
        for pid in list(self.active_players.keys()):
            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(detections):
                if idx in used:
                    continue
                iou = self.IOU(det, self.active_players[pid])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= self.best_match_iou and best_idx != -1:
                matched[pid] = detections[best_idx]
                used.append(best_idx)
            else:
                # Move to inactive
                self.inactive_players[pid] = {
                    "box": self.active_players[pid],
                    "signature": self.signature(frame_count, self.active_players[pid], img)
                }

        # Remove matched from active_players
        for pid in list(self.active_players.keys()):
            if pid not in matched:
                del self.active_players[pid]

        # Re-identify inactive players
        new_players = [detections[i] for i in range(len(detections)) if i not in used]
        reassigned = {}

        for bx in new_players:
            bx_sig = self.signature(frame_count, bx, img)
            best_pid = None
            best_dist = float('inf')

            for pid, info in self.inactive_players.items():
                dist = self.sign_match(bx_sig, info["signature"])
                if dist < best_dist and dist < self.signature_threshold:
                    best_dist = dist
                    best_pid = pid

            if best_pid is not None:
                matched[best_pid] = bx
                reassigned[tuple(bx)] = best_pid # Use tuple as key

        # Remove reactivated players from inactive
        for bx_tuple, pid in reassigned.items():
            # Convert tuple back to numpy array for comparison if needed, or just check pid
            if pid in self.inactive_players:
                 del self.inactive_players[pid]

        # Assign new IDs to remaining unmatched players
        for bx in new_players:
            is_reassigned = False
            for reassigned_bx_tuple in reassigned.keys():
                if np.array_equal(bx, np.array(reassigned_bx_tuple)):
                    is_reassigned = True
                    break
            if not is_reassigned:
                matched[self.id] = bx
                self.id += 1

        # Update active_players
        self.active_players = matched

    def draw_tracks(self, img):
        for pid, box in self.active_players.items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img