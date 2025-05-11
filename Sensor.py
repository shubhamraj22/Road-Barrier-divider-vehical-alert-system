import cv2
import numpy as np
import threading
from playsound import playsound

# Load YOLOv3 weights and config (make sure these files exist in your folder)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load sound alarm
def play_alarm():
    playsound("alarm.wav")  # Place alarm.wav in your project folder

# Estimate distance (you can calibrate this better using real-world testing)
def estimate_distance(box_width):
    return round(5000 / box_width, 2) if box_width != 0 else 9999

# Distance threshold
DISTANCE_THRESHOLD = 10  # Inches

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                center_x, center_y, w, h = (det[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Ensure indexes is iterable
    if len(indexes) > 0:
        for i in indexes.flatten():  # Flattening the result of NMSBoxes
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {int(confidences[i] * 100)}%"
            distance = estimate_distance(w)

            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Bottom corners of the box (approximate wheel positions)
            left_wheel = (x + int(0.2 * w), y + h)
            right_wheel = (x + int(0.8 * w), y + h)
            cam_center = (width // 2, height)

            # Red lines to wheels
            cv2.line(frame, cam_center, left_wheel, (0, 0, 255), 2)
            cv2.line(frame, cam_center, right_wheel, (0, 0, 255), 2)

            # Show distance
            cv2.putText(frame, f"{distance} in", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Alarm trigger
            if distance < DISTANCE_THRESHOLD:
                threading.Thread(target=play_alarm).start()

    # Show the resulting frame
    cv2.imshow("Barrier Detection", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
