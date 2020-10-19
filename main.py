import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file("images/rohit.JPG")
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file("images/preeti.jpg")
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
print(faceloc)
cv2.rectangle(imgelon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255,2))

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255,2))

result =face_recognition.compare_faces([encodeelon],encodetest)
facedis = face_recognition.face_distance([encodeelon],encodetest)
print(result)
print(facedis)
cv2.putText(imgtest,f'{result} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255,2))


cv2.imshow("rohit",imgelon)
cv2.imshow("rohit test",imgtest)
cv2.waitKey(0)

