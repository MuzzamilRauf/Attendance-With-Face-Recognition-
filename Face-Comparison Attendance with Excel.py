import face_recognition
import cv2
import numpy as np
import pandas as pd
import openpyxl
from datetime import datetime

# Initialize attendance list to store recognized names
attendance_list = []

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load known images and encodings
bills_image = face_recognition.load_image_file("Resources/Bills1.jpg")
bills_face_encoding = face_recognition.face_encodings(bills_image)[0]

ellon_image = face_recognition.load_image_file("Resources/Ellon1.jpg")
ellon_face_encoding = face_recognition.face_encodings(ellon_image)[0]

amjad_image = face_recognition.load_image_file("Resources/Amjad.jpg")
amjad_face_encoding = face_recognition.face_encodings(amjad_image)[0]

meer_image = face_recognition.load_image_file("Resources/Meer.jpg")
meer_face_encoding = face_recognition.face_encodings(meer_image)[0]

rashid_image = face_recognition.load_image_file("Resources/Rashid.jpg")
rashid_face_encoding = face_recognition.face_encodings(rashid_image)[0]

muzzamil_image = face_recognition.load_image_file("Resources/My.jpg")
muzzamil_face_encoding = face_recognition.face_encodings(muzzamil_image)[0]

# Known face encodings and their names
known_face_encodings = [
    bills_face_encoding,
    ellon_face_encoding,
    amjad_face_encoding,
    meer_face_encoding,
    rashid_face_encoding,
    muzzamil_face_encoding
]
known_face_names = [
    "Bill-Gates",
    "Ellon-Musk",
    "Amjad",
    "Meer",
    "Rashid",
    "Muzzamil"
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Create an empty DataFrame to store attendance
attendance_df = pd.DataFrame(columns=["Name", "Time"])

# Generate a unique filename using the current date and time
now = datetime.now()
date_time_str = now.strftime("%Y%m%d_%H%M%S")
filename = f"Attendance_{date_time_str}.xlsx"

while True:
    # Capture video frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert frame from BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find faces and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the face with the smallest distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            # If the person is recognized and not already in the attendance list, log attendance
            if name != "Unknown" and name not in attendance_list:
                attendance_list.append(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                # Add the recognized name and time to the DataFrame
                attendance_df = attendance_df._append({"Name": name, "Time": current_time}, ignore_index=True)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Show video
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write the attendance DataFrame to an Excel file with a unique name
attendance_df.to_excel(f"C:/Users/muzammil/PycharmProjects/Face Recognition/{filename}", index=False)

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
