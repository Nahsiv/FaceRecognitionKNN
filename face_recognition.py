# Recognise Faces using some classification algorithm ; Like Logistic, KNN, SVM

# 1. Prepare the data
# 3. Load the training data (numpy arrays of all the person)
# x-values are stored in the numpy arrays
# y-values we need to assign for each person
# 1. Read the video stream using opencv
# 2. extract faces out of it (Testing-Purpose)
# 4. use knn to find the prediction of face(int)
# 5. map the predicted id to name of the user
# 6. Display the predictions on the screen-bounding box and name

# KNN Algorithm


import cv2
import numpy as np
import os


# Write the KNN Function4
# Euclidean Distance
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dis = []
    for i in range(train.shape[0]):
        # Calculate distance and label
        ix = distance(train[i, :-1], test)  # distance
        iy = train[i, -1]  # Label
        dis.append([ix, iy])
    # Sort on the Basis of Distance
    dk = sorted(dis, key=lambda f: f[0])[:5]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    # Get the frequencies of each label
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[index][0]


# Capture the Video Frame
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
dataset_path = './data/'
face_data = []
label = []

class_id = 0  # label for the given file
names = {}  # Mapping between id and name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        # Create labels for the class
        target = class_id * np.ones(data_item.shape[0], )
        class_id += 1
        label.append(target)
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainSet = np.concatenate((face_dataset, face_labels), axis=1)
print(trainSet.shape)

# Testing
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        # Get the region of interest
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + offset + w]
        face_section = cv2.resize(face_section, (100, 100))
        # Predicted label out
        out = knn(trainSet, face_section.flatten())
        # Display on the name and rectangle over the frame
        pred_name = names[int(out)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frane", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
