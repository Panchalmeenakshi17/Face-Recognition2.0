from typing import Any

import numpy as np
import face_recognition
import cv2
import csv
from datetime import datetime

from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

print("started")
video_capture = cv2.VideoCapture(0)

meena_img = face_recognition.load_image_file("faces/meenakshi.jpg")
meena_encoding = face_recognition.face_encodings(meena_img)[0]
kunal_img = face_recognition.load_image_file("faces/kunal.jpg")
kunal_encoding = face_recognition.face_encodings(kunal_img)[0]

known_faceEncodings = [meena_encoding, kunal_encoding]
known_face_names = ["meenakshi", "kunal"]

# list of students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# get current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognise faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faceEncodings, face_encoding)
        face_distances: ndarray[Any, dtype[floating[_64Bit] | float_]] | Any = face_recognition.face_distance(
            known_faceEncodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            lnwriter.writerow([name, current_date, now.strftime("%H:%M:%S")])

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# Release video capture and close CSV file
video_capture.release()
f.close()

print("ended")
