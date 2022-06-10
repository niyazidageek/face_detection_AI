import cv2

trained_data = cv2.CascadeClassifier('data/frontal_faces.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_data.detectMultiScale(gray_img)
    for face in face_coordinates:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 255, 0), 2)
    cv2.imshow('slam', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
