import cv2

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change

img = cv2.imread(r"C:\Users\Admin\PycharmProjects\newproject(20\12\Face-and-Emotion-Recognition\Facial-emotion-recognition\faceman.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5)

for x, y, w, h in faces:

    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    faces1 = img[y:y + h, x:x + w]
    height1, width1, size1 = faces1.shape
    height, width, size = img.shape

resized=cv2.resize(img,(int(img.shape[1]/3), int(img.shape[0]/3)))

cv2.imshow("Deteced-face",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()