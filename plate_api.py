import cv2
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

# Load API key
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

url = "https://api.platerecognizer.com/v1/plate-reader/"

# Load model
model = YOLO("best.pt")

cap = cv2.VideoCapture("video.mp4")

line_y = 300

# 🔥 Store detected plates (IMPORTANT)
detected_plates = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    # Draw line
    cv2.line(frame, (0, line_y), (800, line_y), (0, 0, 255), 2)

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_y = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 🔥 ONLY trigger ONCE when crossing line
            if center_y > line_y:

                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                cv2.imwrite("temp.jpg", plate_crop)

                try:
                    with open("temp.jpg", "rb") as image:
                        response = requests.post(
                            url,
                            files={"upload": image},
                            headers={"Authorization": f"Token {API_TOKEN}"}
                        )

                    data = response.json()

                    if data['results']:
                        plate = data['results'][0]['plate']

                        # 🔥 KEY FIX → detect only once
                        if plate not in detected_plates:
                            detected_plates.add(plate)

                            time_now = datetime.now().strftime("%H:%M:%S")

                            print(f"{plate} at {time_now}")

                            # Save to file
                            with open("log.txt", "a") as f:
                                f.write(f"{plate} - {time_now}\n")

                        # Show plate text
                        cv2.putText(frame, plate,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)

                except Exception as e:
                    print("API Error:", e)

    cv2.imshow("ANPR System", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()