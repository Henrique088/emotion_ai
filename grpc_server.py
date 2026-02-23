import grpc
from concurrent import futures
import cv2
import numpy as np
from ultralytics import YOLO
import emotion_pb2
import emotion_pb2_grpc

class EmotionService(emotion_pb2_grpc.EmotionTrackerServicer):
    def __init__(self):
        # Carrega o modelo com os pesos do AffectNet que você treinou
        self.model = YOLO("./runs/detect/psico_ai/affectnet_96px5/weights/best.pt")

    def DetectEmotion(self, request, context):
        try:
            # O 'request.image_data' aqui é o binário puro do Blob/Buffer
            nparr = np.frombuffer(request.image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return emotion_pb2.EmotionResponse()

            # Inferência otimizada (imgsz=96 conforme seu dataset)
            results = self.model.predict(source=frame, imgsz=96, conf=0.4, verbose=False)
            
            response = emotion_pb2.EmotionResponse()
            
            for box in results[0].boxes:
                det = response.detections.add()
                det.label = self.model.names[int(box.cls)]
                det.confidence = float(box.conf)
                # Normalizando coordenadas ou enviando absolutas para o front
                coords = box.xyxy[0].tolist()
                det.x_min, det.y_min, det.x_max, det.y_max = coords
                
            return response
        except Exception as e:
            print(f"Erro no processamento: {e}")
            return emotion_pb2.EmotionResponse()

def serve():
    # Como é para saúde mental (tempo real), usamos múltiplos workers
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    emotion_pb2_grpc.add_EmotionTrackerServicer_to_server(EmotionService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor gRPC de Emoções ativo na porta 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()