import cv2

#videodan yüz algılama için
#vid=cv2.VideoCapture("C:\OpenCV\/testImages\/faces.mp4")

#webcamden yüz algılama için
vid=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("C:\OpenCV\haarCascade\/frontalface.xml")

while 1:
    _,frame=vid.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.6,5)# video için (gray,1.1,2) webcam için (gray,1.6,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("img",frame)
    if cv2.waitKey(5) & 0xFF==ord("q"):
        break
vid.release()
cv2.destroyAllWindows()