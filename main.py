import cv2

trained_data = cv2.CascadeClassifier('data/frontal_faces.xml')

img = cv2.imread('data/istockphoto-805012064-170667a.jpeg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_data.detectMultiScale(gray_img)

for face in face_coordinates:
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), 2)

cv2.imshow('detected_face', img)

cv2.waitKey()