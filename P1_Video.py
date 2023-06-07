from vidgear.gears import CamGear
import numpy as np
import cv2 
import matplotlib.pyplot as plt

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

    if Obs is None:
        break

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


