import cv2
import pathlib
path=pathlib.Path(cv2.__file__).parent.absolute()/"data_haarcascade_frontalface_default.xml"
print(path)
clf= cv2.CascadeClassifier(str(path))
camera = cv2.VideoCapture(0)
while True:
  _, frame = camera.read()
  gray = cv2.cvtcolor(frame, cv2.COLOR_BGRGRAY)
  faces = clf.datectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags =cv2.CASCADE_SCALE_IMAGE )

  for(x,y, width, height) in faces:
    cv2.rectangle(frame,(x,y),(x+width, y+height), (255,255,0),2)

  cv2.imshow("faces", frame)
  if cv2.waitkey(1) == ord("q"):
    break

  camera.release()
  cv2.destroyAllWindows()
  
