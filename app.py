import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import requests

app = Flask(__name__)
# Konfigurasi CORS: izinkan permintaan dari frontend Anda (misalnya localhost:8080 atau domain deploy Anda)
CORS(app, resources={r"/predict": {"origins": "http://localhost:9001"}}) # Ganti dengan origin frontend Anda

# Path ke model .h5 Anda (setelah diupload ke server Flask nanti)
# Di lingkungan development lokal, Anda bisa letakkan model.h5 di sini
# Contoh path:
MODEL_PATH = '/app/model/bag_classifier_best_model.h5'
MODEL = None # Model akan dimuat saat aplikasi dimulai

# Pastikan ukuran gambar yang diharapkan oleh model Anda
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Daftar kelas (sesuaikan dengan urutan output model Anda)
CLASS_NAMES = ['Anorganik', 'Organik']

def download_model_if_not_exists():
    if not os.path.exists(MODEL_PATH):
        print("Model tidak ditemukan. Mengunduh dari Google Drive...")
        file_id = "GANTI_DENGAN_FILE_ID_GOOGLE_DRIVE"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model berhasil diunduh.")
        else:
            print("❌ Gagal mengunduh model. Status code:", response.status_code)
            
def load_model():
    """Memuat model Keras yang sudah terlatih."""
    global MODEL
    if MODEL is None:
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            MODEL.summary() # Untuk debug, pastikan model dimuat dengan benar
            print("Model ML berhasil dimuat!")
        except Exception as e:
            print(f"Error memuat model ML: {e}")
            MODEL = None # Pastikan MODEL tetap None jika gagal

def preprocess_image(image):
    """Preprocessing gambar sesuai kebutuhan model."""
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0  # Normalisasi ke [0, 1]
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model ML belum dimuat. Coba lagi nanti.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar ditemukan.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file gambar yang dipilih.'}), 400

    if file:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Pastikan RGB
            processed_image = preprocess_image(image)

            predictions = MODEL.predict(processed_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index]) * 100

            return jsonify({
                'class_name': predicted_class_name,
                'confidence': f"{confidence:.2f}%"
            })
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {e}'}), 500
    return jsonify({'error': 'Permintaan tidak valid.'}), 400

# Endpoint untuk kesehatan (opsional, untuk memastikan server berjalan)
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None}), 200

# Memuat model saat aplikasi Flask dimulai
# Gunakan app.before_first_request jika ingin hanya sekali pada request pertama
with app.app_context():
    download_model_if_not_exists()
    load_model() # Muat model saat aplikasi Flask dimulai


if __name__ == '__main__':
    # Untuk development lokal:
    # pip install python-dotenv
    # Di file .env di root ml-server: FLASK_ENV=development, FLASK_APP=app.py
    # flask run
    # Atau di Railway, server akan otomatis menjalankan app.py
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000),debug=True)