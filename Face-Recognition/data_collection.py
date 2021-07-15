import cv2
import numpy as np

#initialize webcom
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("./OpenCV Basics/haarcascade_frontalface_alt.xml")
face_data = []
datapath = "./data/"
filename = input("Enter your name: ")
while True:
    ret, frame = cap.read()
    

    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)#5-> no of neighbours
    if len(faces)==0:
        continue

    faces = sorted(faces, key = lambda f:f[2]*f[3])
    #now picking the face with largest area
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 3)

        #cropping face area
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+offset+w]
        face_section = cv2.resize(face_section, (100, 100))
        face_data.append(face_section)

    cv2.imshow("Frame",frame)
    #cv2.imshow("Gray_Frame",gray_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

np.save(datapath + filename + '.npy', face_data)
print("Data Saved")

cap.release()
cv2.destroyAllWindows()   
