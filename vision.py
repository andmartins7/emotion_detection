from tkinter import filedialog
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Importar o modelo VGG pré-treinado para detecção de emoções
from keras.models import load_model

# Carregar o modelo VGG pré-treinado
emotion_model = load_model('emotion_model.h5')

# Dicionário para mapear os rótulos de emoção para suas respectivas classes
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Dicionário de cores fixas para cada emoção
emotion_colors = {
    'Angry': 'red',
    'Disgust': 'purple',
    'Fear': 'orange',
    'Happy': 'green',
    'Sad': 'blue',
    'Surprise': 'pink',
    'Neutral': 'gray'
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = None
fig = None
ax = None
emotions = []
probabilities = []


def process_frame(frame):
    global emotions, probabilities

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotions = []
    probabilities = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_roi = cv2.resize(face_roi, (64, 64))
        normalized_roi = resized_roi / 255.0
        reshaped_roi = np.reshape(normalized_roi, (1, 64, 64, 1))

        # Classificação da emoção usando o modelo VGG
        emotion_prediction = emotion_model.predict(reshaped_roi)
        emotion_indices = np.argsort(emotion_prediction[0])[::-1]

        for idx in emotion_indices:
            emotion_text = emotion_labels[idx]
            emotion_probability = emotion_prediction[0, idx]
            emotions.append(emotion_text)
            probabilities.append(emotion_probability)

        # Desenha um retângulo e exibe a emoção de maior probabilidade no quadro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{emotions[0]}: {probabilities[0]*100:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

    update_emotion_chart()

    return frame


def process_video():
    global cap
    path_video = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4")])
    if len(path_video) > 0:
        cap = cv2.VideoCapture(path_video)
        visualize(cap)


def process_webcam():
    global cap
    cap = cv2.VideoCapture(0)
    visualize(cap)


def finalizar_limpar(video_window=None):
    global cap
    if cap is not None:
        cap.release()
    video_window.destroy()


def visualize(cap):
    video_window = Toplevel()
    video_window.title("Visualização de Vídeo/Webcam")
    video_window.protocol("WM_DELETE_WINDOW", finalizar_limpar)

    lblVideo = Label(video_window)
    lblVideo.pack()

    btnEnd = Button(video_window, text="Finalizar visualização e limpar", command=lambda: finalizar_limpar(video_window))
    btnEnd.pack(pady=10)

    def update_video():
        nonlocal cap
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im.resize((640, 480)))  # Ajuste o tamanho da imagem aqui

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.update()

            video_window.after(1, update_video)
        else:
            cap.release()
            video_window.destroy()

    update_video()


def update_emotion_chart():
    global fig, ax, emotions, probabilities

    if emotions:
        ax.clear()

        x = np.arange(len(emotions))
        colors = [emotion_colors.get(emotion, 'gray') for emotion in emotions]
        ax.barh(x, probabilities, color=colors)

        for i, (emotion, probability) in enumerate(zip(emotions, probabilities)):
            ax.text(0.1, i, f'{emotion}: {probability*100:.2f}%', fontsize=12, color=colors[i])

        ax.set_xlim([0, 1])
        ax.set_yticks(x)
        ax.set_yticklabels(emotions, rotation=0)
        ax.set_xlabel('Probability')
        ax.set_title('Emotion Probability')

        plt.pause(0.001)


root = Tk()
root.title("Detector de Emoções")  # Definir o título da janela

lblInfo1 = Label(root, text="DETECÇÃO DE EMOÇÕES", font="bold")
lblInfo1.grid(column=0, row=0, columnspan=2)

btnVideo = Button(root, text="Processar vídeo", command=process_video)
btnVideo.grid(column=0, row=1, pady=10)

btnWebcam = Button(root, text="Processar webcam", command=process_webcam)
btnWebcam.grid(column=1, row=1, pady=10)

# Configuração do gráfico
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
plt.tight_layout()

root.mainloop()