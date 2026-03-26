import cv2
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

# API URL
url = "https://api.platerecognizer.com/v1/plate-reader/"

# Open video file
cap = cv2.VideoCapture("video.mp4")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # 🔥 Process every 40th frame (reduce API calls)
    if frame_count % 40 != 0:
        continue

    # 🔥 Crop center region (focus on vehicles)
    h, w, _ = frame.shape
    crop = frame[int(h*0.4):int(h*0.8), int(w*0.2):int(w*0.8)]

    # Save cropped frame
    cv2.imwrite("temp.jpg", crop)

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
            print("Detected Plate:", plate)

            # Draw rectangle on original frame
            cv2.rectangle(frame,
                          (int(w*0.2), int(h*0.4)),
                          (int(w*0.8), int(h*0.8)),
                          (0,255,0), 2)

            cv2.putText(frame, plate, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    except Exception as e:
        print("Error:", e)

    # Show video
    cv2.imshow("CCTV ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()