import cv2
# openned camera
camera = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#detector = cv2.CascadeClassifier('dogscascade.xml')

# checking if camera is openned
if not camera.isOpened():
    print("Cannot open camera")
    exit()
# camera watching loop
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    bboxes = detector.detectMultiScale(frame)
     
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255), 2)
    # Display the resulting frame
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
