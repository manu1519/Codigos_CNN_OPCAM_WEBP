from vidgear.gears import CamGear
import numpy as np
import cv2
import matplotlib.pyplot as plt

opt = {
    "CAP_PROP_FRAME_WIDTH": 720,
    "CAP_PROP_FRAME_HEIGHT": 60,
    "CAP_PROP_FPS": 80,
}

#for i in range(4):
Obs = CamGear(source=1, logging=True, colorspace=None, **opt).start()

while True: 
    frames = Obs.read()

    if Obs is None:
        break

    cv2.imshow('Salida', frames)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
Obs.stop()
