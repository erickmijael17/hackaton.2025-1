import os
import numpy as np
from tensorflow.keras.models import load_model
from utils import cargar_imagen, mostrar_resultado  # 👈 Usamos las funciones de utils

MODEL_PATH = "../modelo/modelo_final.h5"
CLASSES = ['textiles', 'ceramicas', 'madera']

def predecir(ruta_img):
    modelo = load_model(MODEL_PATH)
    img_array = cargar_imagen(ruta_img)
    prediccion = modelo.predict(img_array)[0]
    indice = np.argmax(prediccion)
    clase = CLASSES[indice]
    prob = prediccion[indice] * 100

    print(f"\n🖼 Imagen: {os.path.basename(ruta_img)}")
    print(f"🔎 Predicción: {clase} ({prob:.2f}%)")

    mostrar_resultado(ruta_img, clase, prob)

if __name__ == "__main__":
    ruta = input("🔍 Ruta de la imagen a clasificar: ")
    if os.path.exists(ruta):
        predecir(ruta)
    else:
        print("❌ Imagen no encontrada.")
