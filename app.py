from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing import image  # Usar las mismas funciones de Keras que en el notebook
import io
import tensorflow as tf

app = Flask(__name__, template_folder="")  # Cambiado para buscar en la raíz del proyecto

# Cargar el modelo entrenado desde el archivo
model = tf.keras.models.load_model('dog_cat_classifier.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Usar el mismo método de Keras para cargar la imagen desde los datos del archivo
    img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra
    img_array /= 255.0  # Normalizar la imagen

    # Hacer la predicción
    pred = model.predict(img_array)
    prediction = "Perro" if pred[0] > 0.5 else "Gato"

    # Retornar la predicción como respuesta JSON
    return jsonify({'class': prediction})

if __name__ == '__main__':
    app.run(debug=True)
