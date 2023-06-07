ROBOFLOW_API_KEY = "pBWT53OXypNO7EkbVu33"
ROBOFLOW_MODEL = "titi-uvqwk"
ROBOFLOW_SIZE = 416

import cv2
import base64
import numpy as np
import requests
import time

upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?access_token=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

video = cv2.VideoCapture(0)

if (video.isOpened()==False):
    print("Error")

def infer():
    ret, img = video.read()

    if ret == True:
        height, width, channels = img.shape
        scale = ROBOFLOW_SIZE / max(height, width)
        img = cv2.resize(img, (round(scale*width), round(scale*height)))
        
        retval, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer)

        # Get prediction from Roboflow Infer API
        resp = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }, stream=True).raw

        # Parse result image
        
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        print(image)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print(image)

    return image
        

# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    ima = infer()
    # And display the inference results
    cv2.imshow('image', ima)

# Release resources when finished
video.release()
cv2.destroyAllWindows()
