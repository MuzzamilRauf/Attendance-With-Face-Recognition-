import cv2
import face_recognition

import numpy as np

# Load and process the stored image (to be matched)
imgElon = face_recognition.load_image_file("Resources/My.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
faceLocation = face_recognition.face_locations(imgElon)[0]
faceEncode = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocation[3],faceLocation[0]), (faceLocation[1], faceLocation[2]), (255,0,255), 3)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame to RGB (face_recognition uses RGB while OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Compare the faces in the current frame to the stored image
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the current face with the stored one
        result = face_recognition.compare_faces([faceEncode], face_encoding)
        distance = face_recognition.face_distance([faceEncode], face_encoding)

        # Scale the face location back to the original frame size
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face in the webcam frame
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)


        # Add the result (Match or not) and the distance on the frame
        cv2.putText(frame, f'{result[0]}, {round(distance[0], 2)}', (left, top - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 255), 1)



    # Display the webcam frame
    cv2.imshow("Webcam", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
