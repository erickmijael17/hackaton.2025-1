import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def cargar_imagen(ruta_img, target_size=(224, 224)):
    img = image.load_img(ruta_img, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def mostrar_resultado(ruta_img, clase, prob):
    img = image.load_img(ruta_img)
    plt.imshow(img)
    plt.title(f"{clase} ({prob:.2f}%)")
    plt.axis("off")
    plt.show()
