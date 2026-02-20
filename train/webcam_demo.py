import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from model import get_model
from collections import deque, Counter

# --- CONFIGURAÇÕES E PASTAS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpresa', 'Neutro']
base_coleta = "data/coleta_feedback"
for cls in CLASSES: os.makedirs(os.path.join(base_coleta, cls), exist_ok=True)

# --- CARREGAR DETECTOR DE ROSTO (OpenCV DNN) ---
# Caminho absoluto baseado no local deste script
BASE_DIR = r"C:\Users\henri\Documents\ia"
PROTOTXT = os.path.join(BASE_DIR, "models", "deploy.prototxt")
MODEL_DNN = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

# Debug visual para conferência
print(f"DEBUG - Caminho Prototxt: {PROTOTXT}")
print(f"DEBUG - Caminho Modelo: {MODEL_DNN}")

if not os.path.exists(PROTOTXT):
    # Se ele ainda der erro, tenta procurar pela versão .txt que o Windows cria
    PROTOTXT_ALT = PROTOTXT + ".txt"
    if os.path.exists(PROTOTXT_ALT):
        PROTOTXT = PROTOTXT_ALT
        print("Aviso: Usando versão .txt encontrada")
    else:
        raise FileNotFoundError(f"Não achei o arquivo em: {PROTOTXT}")

face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_DNN)
print("Detector carregado com sucesso!")

# --- CARREGAR MODELO DE EMOÇÃO ---
emotion_model = get_model().to(device)
emotion_model.load_state_dict(torch.load("../models/SOTA_FER2013.pt", map_location=device, weights_only=True))
emotion_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

history = deque(maxlen=10)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # Detecção de Rosto via DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5: # Filtro de detecção
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            
            # Garantir que o recorte está dentro da imagem
            x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)
            face_roi = frame[y:y1, x:x1]

            if face_roi.size > 0:
                # Converter para RGB e inferir emoção
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                input_tensor = transform(face_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = emotion_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf_emocao, pred = torch.max(probs, 1)
                    
                    history.append(pred.item())
                    label = CLASSES[Counter(history).most_common(1)[0][0]]
                    conf_val = conf_emocao.item() * 100

                # --- LÓGICA DE APRIMORAMENTO (ACTIVE LEARNING) ---
                if conf_val < 45.0: # Salva rostos que o modelo teve dúvida
                    fname = f"{base_coleta}/{label}/duvida_{np.random.randint(10000)}.jpg"
                    cv2.imwrite(fname, face_roi)

                # Desenhar na tela
                color = (0, 255, 0) if label == 'Feliz' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                cv2.putText(frame, f"{label} ({conf_val:.1f}%)", (x, y-10), 2, 0.7, color, 2)

    cv2.imshow('Analise de Emocao Robusta', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()