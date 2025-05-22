import cv2
import face_recognition
from face_recognition import face_encodings

import face_recognition_models
import numpy as np

imgElon = face_recognition.load_image_file("Resources/Bills1.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
faceLocation = face_recognition.face_locations(imgElon)[0]
faceEncode = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocation[3],faceLocation[0]), (faceLocation[1], faceLocation[2]), (255,0,255), 3)

# imgElonMusk = face_recognition.load_image_file("Resources/Elon-Musk.jpg")
# imgElonMusk = cv2.cvtColor(imgElonMusk, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("Resources/EllonTest.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
faceLocationT = face_recognition.face_locations(imgTest)[0]
faceEncodeT = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocationT[3],faceLocationT[0]), (faceLocationT[1], faceLocationT[2]), (255,0,255), 3)

result = face_recognition.compare_faces([faceEncode], faceEncodeT)
distance = face_recognition.face_distance([faceEncode], faceEncodeT)
print(result[0], distance)
cv2.putText(imgTest, f'{result[0]}, {round(distance[0], 2)}', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 255), 1)

cv2.imshow("ELLON", imgElon)
# cv2.imshow("ELLON TEST", imgElonMusk)
cv2.imshow("Elolon Test", imgTest)

cv2.waitKey(0)

cv2.destroyAllWindows()
