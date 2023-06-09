const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');

const app = express();
const upload = multer();

// Load model TensorFlow.js
const modelPath = 'saved_model_js/model.json';
const loadModel = async () => {
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  return model;
};

// Fungsi untuk meresize gambar menggunakan tf.image.resizeBilinear
const resizeImage = async (imageData, width, height) => {
  const resizedTensor = tf.image.resizeBilinear(imageData, [width, height]);
  return resizedTensor;
};

// Fungsi deteksi gambar
const detectImage = async (imageData) => {
  try {
    // Lakukan prediksi menggunakan model
    const model = await loadModel();
    const prediction = model.predict(imageData);

    // Dapatkan indeks kelas prediksi dengan nilai tertinggi
    const predictedClassIndex = prediction.argMax(1).dataSync()[0];

    // Tentukan teks prediksi berdasarkan indeks kelas
    let predictionText;
    if (predictedClassIndex === 0) {
      predictionText = 'Black Pod Rot';
    } else if (predictedClassIndex === 1) {
      predictionText = 'Healthy';
    } else if (predictedClassIndex === 2) {
      predictionText = 'Monilia';
    } else {
      predictionText = 'Unknown';
    }

    return predictionText;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};

// API endpoint untuk prediksi gambar
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    // Ambil data gambar dari form-data
    const imageBuffer = req.file.buffer;

    // Konversi data gambar menjadi tensor dan lakukan normalisasi
    const imageData = tf.node.decodeImage(imageBuffer);
    const normalizedImageData = imageData.toFloat().div(255).expandDims();

    // Resize gambar menjadi 250x250
    const resizedImageData = await resizeImage(normalizedImageData, 250, 250);

    // Deteksi gambar dan dapatkan hasil prediksi
    const prediction = await detectImage(resizedImageData);

    // Bebaskan memori yang digunakan oleh tensor
    imageData.dispose();
    normalizedImageData.dispose();
    resizedImageData.dispose();

    // Kembalikan hasil prediksi sebagai respons
    res.json({ prediction });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Jalankan server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server listening on port http://localhost:${PORT}`);
});
