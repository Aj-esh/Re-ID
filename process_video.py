import cv2

def process_video(path, model, tracker):
    cap = cv2.VideoCapture(path)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        result = model(frame)[0]
        img = result.orig_img
        detections = [box[:4].cpu().numpy() for box in result.boxes.data if int(box[-1]) > 0]

        tracker.update(detections, frame_idx, img)
        tracked_img = tracker.draw_tracks(img.copy())

        # Display
        cv2.imshow("Tracked Frame", tracked_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1  # frame_idx increment

    cap.release()
    cv2.destroyAllWindows()
