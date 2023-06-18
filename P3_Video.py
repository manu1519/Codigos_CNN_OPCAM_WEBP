from vidgear.gears import CamGear
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from keras import models

model = models.load_model(r'save_NH_3/CCNN.h5')
# Se configuran los parámetros de la pantalla
# a mostrar para el operador.
opt = {
    "CAP_PROP_FRAME_WIDTH": 1080,
    "CAP_PROP_FRAME_HEIGHT": 60,
    "CAP_PROP_FPS": 80,
}

# Se realiza un escaneo de puertos USB
# en el dispositivo, para localizar la señal
Obs = CamGear(source=1, logging=True, colorspace=None, **opt).start()




#Se realiza la ventana para previsualizar el stream
while True:
    frames = Obs.read()

    im = Image.fromarray(frames, 'RGB')

    #Resizing into 128x128 because we trained the model with this image size.
    im = im.resize((180,180))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)

    if Obs is None:
        break

    prediction = int(model.predict(img_array)[0][0])

    if prediction == 0:
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Salida',frames)

    # Se crea una ruta para terminar la sesión
    # desde el CGS marcada por el operador
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


# Se cierra la ventana para visualizar y se cierra
# el uso del dispositivo (camara)
cv2.destroyAllWindows()
Obs.stop()

