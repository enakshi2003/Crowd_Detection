import torch
import cv2
import numpy as np

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('Frame')

cap = cv2.VideoCapture('vid1.mp4')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    roi = frame[1:495, 1:1015]
    results = model(roi)

    # Initialize a counter for rectangles in each frame
    frame_rectangle_count = 0

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        
        # Draw rectangles on the ROI image
        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Increment the frame-level rectangle count
        frame_rectangle_count += 1

    # Display the frame-level rectangle count on the video frame
    cv2.putText(frame, f'PEOPLE: {frame_rectangle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
