# Python Script that capture images from the webCam Video Stream(0)
# Extract all Faces from the image frame (using Har cascades)
# Store the faces into numpy arrays

# 1. Read and show video stream and capture images
# 2. Detect Faces and show Bounding Box
# 3. Flatten the Largest Face and store it to the numpy arrays
# 4. Repeat the above for multiple person to generate the training data

import cv2
import numpy as np

# Initialize the camera and capture the video
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the Person")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # print(faces)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    # Pick the largest face that is the based on sorting
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        #     Extract the region of Interest(Crop out the required Face
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
            cv2.imshow("Face_Section", face_section)
        skip += 1
    cv2.imshow("Frame", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert our face list into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save this data into the file system
np.save(dataset_path+file_name+'.npy', face_data)
print("data saved successfully "+dataset_path+file_name+".npy")
cap.release()
cv2.destroyAllWindows()
