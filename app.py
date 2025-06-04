import os
import cv2
import numpy as np
from keras.models import load_model

CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'emotion_detection_model.h5'

# Dicionário com as emoções correspondentes aos índices
EMOTION_DICT = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Fearful',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprised',
}


def verify_file(path: str) -> None:
    """Verifica se o arquivo existe."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Arquivo não encontrado: {path}')


def load_models():
    """Carrega e valida os modelos de detecção."""
    verify_file(CASCADE_PATH)
    verify_file(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f'Erro ao carregar cascata facial: {CASCADE_PATH}')
    model = load_model(MODEL_PATH)
    return face_cascade, model


def main() -> None:
    face_cascade, model = load_models()
    cap = cv2.VideoCapture(0)  # Acessa a câmera de vídeo
    if not cap.isOpened():
        raise RuntimeError('Não foi possível acessar a câmera.')

    while True:
        ret, frame = cap.read()  # Captura um quadro de vídeo
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        predictions = []

        for i, (x, y, w, h) in enumerate(faces):
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            preds = model.predict(roi_gray)[0]
            label = EMOTION_DICT[np.argmax(preds)]
            predictions.append((preds, label))
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'Face {i}',
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow('frame', frame)
        janela = np.zeros((500, 250, 3), np.uint8)
        cv2.putText(
            janela,
            'Índices das faces detectadas',
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        for i, (x, y, w, h) in enumerate(faces):
            preds, label = predictions[i]
            color = (
                (0, 0, 255)
                if label in ['Angry', 'Disgusted', 'Fearful', 'Sad']
                else (0, 255, 0)
            )
            cv2.putText(
                janela,
                'Face {0} - {1}: {2:.2f}%'.format(i, label, preds[np.argmax(preds)] * 100),
                (10, 30 + (i * 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.imshow('Indices das faces detectadas', janela)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

