# Librerías arbir camara y detección
from vidgear.gears import CamGear
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from keras import models
# Librerías transmisión
from flask import Flask
from flask import render_template
from flask import Response
import cv2

app = Flask(__name__,template_folder='temp')

def generate():
    model = models.load_model(r'save_NH_2/CCNN.h5')
    Obs = CamGear(source=0, logging=True, colorspace=None).start()
    while True:
        frames = Obs.read()

        im = Image.fromarray(frames, 'RGB')

    #Resizing into 180x180 because we trained the model with this image size.
        im = im.resize((180,180))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)

        if Obs is None:
            break

        prediction = int(model.predict(img_array)[0][0])

        if prediction == 0:
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            (flag,encod) = cv2.imencode(".jpg", frames)

        (flag,encod) = cv2.imencode(".jpg", frames)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encod) + b'\r\n')
        #cv2.imshow('Salida',frames)


# Se cierra la ventana para visualizar y se cierra
# el uso del dispositivo (camara)
    cv2.destroyAllWindows()
    Obs.stop()



@app.route("/video")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype= "multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__":
    app.run(debug=True)