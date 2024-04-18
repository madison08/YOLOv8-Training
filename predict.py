import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('/Users/madisonattobra/Documents/data', 'videos')

# video_path = os.path.join(VIDEOS_DIR, 'Enregistrement de l’écran 2024-04-17 à 18.45.51.mp4') 
video_path = os.path.join(VIDEOS_DIR, 'VID-20240415-WA0001.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    print(f"Detection results: {results.boxes.data.tolist()}")  # print detection results


    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        print(f"Score: {score}")  # print detection score


        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()