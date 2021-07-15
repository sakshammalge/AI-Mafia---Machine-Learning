#Run the data_collection program first to store the data of your face.
#Then run the face_recognition program 
import cv2
import numpy as np
import os

def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(train, test, k=5 ):

    distance = []
    for i in range(train.shape[0]):
        x = train[i, :-1]
        y = train[i,-1]

        d = dist(x, test)
        distance.append((d,y))

    distance = sorted(distance)
    vals = distance[:k]
    vals = np.array(vals)

    out = np.unique(vals[1], return_counts=True)
    index = np.argmax(out[1])
    prediction = out[0][index]

    return prediction

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./OpenCV Basics/haarcascade_frontalface_alt.xml")

id = 0
names = {}
datapath = "./data/"
face_data = []
labels = []

for fx in os.listdir(datapath):
    if fx.endswith(".npy"):
        names[id] = fx[:-4]

        data = np.load(datapath+fx)
        face_data.append(data)

        outputid = id * np.ones((data.shape[0],))
        id+=1
        labels.append(outputid)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1,1))
training_set = np.concatenate((face_dataset, face_labels),axis = 1)

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
        try:
            face_section = cv2.resize(face_section, (100, 100))
        except:
            continue
        out = knn(training_set, face_section.flatten())        
        name = names[int(out)]
        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),2, cv2.LINE_AA)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()  
