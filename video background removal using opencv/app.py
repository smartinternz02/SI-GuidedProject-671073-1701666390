
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
cap = cv2.VideoCapture('Data/Sample Video.mp4')

def gen_frames():
    background = cv2.imread("office.jpg")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channel = frame.shape
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentation.process(RGB)
        mask = results.segmentation_mask

        rsm = np.stack((mask,)*3, axis=-1)
        condition = rsm > 0.6
        condition = np.reshape(condition, (height, width, 3))
        background = cv2.resize(background, (width, height))
        output = np.where(condition, frame, background)
        _, buffer = cv2.imencode('.jpg', output)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        uploaded_file = request.files['video']
        if uploaded_file.filename != '':
           

            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

   
    return render_template('index.html')


@app.route('/video_processed')
def video_processed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
