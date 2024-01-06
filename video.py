import cv2

cap = cv2.VideoCapture(0)
# 0 indicates that your default webcam will be used

while True:
    ret, fram = cap.read()
    if not ret:
        continue
    gray_frame = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video Frame", fram)
    cv2.imshow("Gray_Frame",gray_frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
