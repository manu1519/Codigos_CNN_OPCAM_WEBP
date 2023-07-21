# Codigos_CNN_OPCAM_WEBP

Se presenta un listado con los diferentes códigos que se implementaron para realizar la detección de personas desde 7 - 30 m de altura, también se pueden mostrar códigos para abrir cualquier cámara por medio de la librería VidGear y por último un enlace entre servidor y usuario de tipo local entre la Raspberry Pi y cualquier dispositivo dentro del mismo localhost ocupando Flask

1.- CAM_OP.py  ->  Se ocupa para poder mostrar video de cualquier cámara conectada por USB dentro de la librería l4tb 

2.- CCNN.py  ->  Se ocupa para realizar la red neuronal convolucional por medio del uso de Keras y Tensorflow

3.- COMON.py  ->  Hace la conexión entre usuario y servidor por medio del localhost con la librería Flask, a la par se ocupa para mostrar la transmisión de video ocupando OpenCV.

4.- CT.py  ->  Es un código experimental para la comunicación entre un servidor web llamado Roboflow con cualquier usuario por medio de Python

5.- MAPIMA.py  ->  Busca mostrar el video que se obtiene por OpenCV e implementa por medio del archivo obtenido en CCNN.py una detección si cumple los parámetros requeridos, como son la rotación, inclinación y altura de la cámara. Para observar una detección el video mostrado cambiará de color GRAY a RGB

6.- P#_Video.py -> Son los intentos para cambiar las rotaciones de las imágenes que genera el programa de CCNN.py para encontrar la rotación correcta.

7.- PT_Video.py -> Es el resultado final encontrando la inclinación y rotación correcta.

