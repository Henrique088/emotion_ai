import cv2
import torch
import numpy as np

# Labels padrão de emoção
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ----- 1) Carregar detector de rosto (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----- 2) Carregar modelo pré-treinado (placeholder por enquanto)
# depois vamos trocar por um modelo real FER/AffectNet
class DummyEmotionModel:
    def predict(self, face_tensor):
        probs = torch.rand(len(EMOTIONS))
        probs = probs / probs.sum()
        idx = torch.argmax(probs)
        return EMOTIONS[idx], probs[idx].item()

model = DummyEmotionModel()

# ----- 3) Função principal
def detect_emotion(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Erro ao carregar imagem.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("Nenhum rosto detectado.")
        return

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        # preprocessamento padrão
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0

        face_tensor = torch.tensor(face_normalized).float()
        face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)

        emotion, confidence = model.predict(face_tensor)

        print(f"Emoção detectada: {emotion}")
        print(f"Confiança: {confidence:.2f}")

        # desenhar no frame
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            image,
            f"{emotion} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----- EXECUÇÃO
detect_emotion("sample.jpg")
