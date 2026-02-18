import torch
from preprocess import transform
from emotion_model import load_model, EMOTIONS

model = load_model()

def predict_emotion(face_img):

    tensor = transform(face_img)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

    emotion = EMOTIONS[predicted.item()]
    confidence = confidence.item()

    return emotion, confidence
