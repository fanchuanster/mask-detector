from PIL import Image

from model import MODEL_FILE, get_transformer, build_model
import torch
import cv2
from face import Face

model = build_model()
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()
model.cpu()

def preprocess(frame):
    image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_array)
    transformed_image = image_transforms(pil_image)
    img = transformed_image.unsqueeze_(0)
    img = transformed_image.float()
    return transformed_image

def predict(transformed_image):
    output = model(transformed_image)
    _, predicted = torch.max(output, 1)
    classification = predicted.data[0]
    print("classification", classification)
    index = int(classification)
    return index

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2
image_transforms = get_transformer()
#faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    transformed_image = preprocess(frame)
    label = predict(transformed_image)
    print(label)

    if(label == 1):
        label = "masked"
    else:
        label = "no mask"
    cv2.putText(frame, label, (100, height - 20),
                font, 1,
                (255, 0, 0) if label == 0 else (0, 255, 255),
                1,cv2.LINE_AA)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()