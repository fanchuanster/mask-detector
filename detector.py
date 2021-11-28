from model import MODEL_FILE, build_model
import torch
import cv2

model = build_model()
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2
#faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    label = classify_face(frame)
    if(label == 'with_mask'):
        print("No Beep")
    else:
        sound.play()
        print("Beep")
    cv2.putText(frame,str(label),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()