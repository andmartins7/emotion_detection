import cv2
import numpy as np
from keras.models import load_model


def main():
    file_path = 'haarcascade_frontalface_default.xml'

    # Carrega o modelo de detecção de faces
    face_cascade = cv2.CascadeClassifier(file_path)

    emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral',
                    5: 'Sad', 6: 'Surprised'}  # Dicionário com as emoções correspondentes aos índices

    file_path_emotion = 'emotion_detection_model.h5'

    # Carrega o modelo de detecção de emoções
    model = load_model(file_path_emotion)

    cap = cv2.VideoCapture(0)  # Acessa a câmera de vídeo
    while True:
        ret, frame = cap.read()  # Captura um quadro de vídeo
        if not ret:
            print('Failed to capture frame from camera. Exiting.')
            break
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detecta faces no quadro

        for i, (x, y, w, h) in enumerate(faces):
            # Extrai a região de interesse correspondente ao rosto
            roi_gray = gray[y:y+h, x:x+w]
            # Redimensiona a imagem para o tamanho esperado pelo modelo de detecção de emoções
            roi_gray = cv2.resize(roi_gray, (64, 64))
            # Normaliza os valores de pixel para o intervalo [0, 1]
            roi_gray = roi_gray.astype('float')/255.0
            # Adiciona uma dimensão ao início do array para torná-lo compatível com o modelo
            roi_gray = np.expand_dims(roi_gray, axis=0)
            # Adiciona uma dimensão ao final do array para torná-lo compatível com o modelo
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            preds = model.predict(roi_gray)[0]  # Executa a predição de emoção
            # Determina a emoção correspondente ao índice com maior probabilidade
            label = emotion_dict[np.argmax(preds)]
            # Exibe a emoção detectada acima do retângulo que envolve o rosto
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Desenha um retângulo ao redor do rosto detectado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Exibi o índice na parte inferior do retângulo
            cv2.putText(frame, f'Face {i}', (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)  # Exibe o quadro de vídeo
        # Cria uma janela para exibir os índices das faces detectadas
        janela = np.zeros((500, 250, 3), np.uint8)
        cv2.putText(janela, 'Índices das faces detectadas', (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for i, (x, y, w, h) in enumerate(faces):
            # Determina a emoção correspondente ao índice com maior probabilidade
            label = emotion_dict[np.argmax(preds)]
            if label == 'Angry' or label == 'Disgusted' or label == 'Fearful' or label == 'Sad':
                color = (0, 0, 255)  # Vermelho
            else:
                color = (0, 255, 0)  # Verde

            cv2.putText(janela, 'Face {0} - {1}: {2:.2f}%'.format(i, label, preds[np.argmax(
                preds)] * 100), (10, 30 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Exibe o quadro de índices
        cv2.imshow('Indices das faces detectadas', janela)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Sai do loop quando a tecla 'q' é pressionada
            break

    cap.release()  # Libera a câmera de vídeo
    cv2.destroyAllWindows()  # Fecha as janelas abertas


if __name__ == '__main__':
    main()
