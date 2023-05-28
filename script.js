// Memuat model TensorFlow.js yang sudah jadi
async function loadModel() {
    const model = await tf.loadLayersModel('saved_model_js/model.json');
    return model;
  }
  
  // Memprediksi gambar menggunakan model
  async function predictImage(model, file) {
    try {
      // Mengubah file menjadi objek Image
      const img = new Image();
      img.src = URL.createObjectURL(file);
  
      // Menunggu gambar selesai dimuat
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
  
      // Mengubah gambar menjadi tensor
      const tensor = tf.browser.fromPixels(img).toFloat();
      const resizedTensor = tf.image.resizeBilinear(tensor, [250, 250]);
      const expandedTensor = resizedTensor.expandDims();
  
      // Normalisasi tensor
      const normalizedTensor = expandedTensor.div(255.0);
  
      // Memprediksi kelas gambar
      const predictions = await model.predict(normalizedTensor).data();
  
      // Mengambil label dengan probabilitas tertinggi
      const maxProbIndex = predictions.indexOf(Math.max(...predictions));
  
      // Mengembalikan prediksi label
      return maxProbIndex;
    } catch (error) {
      console.warn('Failed to predict image:', error);
      throw error;
    }
  }
  
  // Menginisialisasi aplikasi
  async function init() {
    const imageUpload = document.getElementById('image-upload');
    const predictButton = document.getElementById('predict-button');
    const predictionContainer = document.getElementById('prediction-container');
  
    // Memuat model
    const model = await loadModel();
  
    // Menghandle event saat tombol "Predict" ditekan
    predictButton.addEventListener('click', async () => {
      const file = imageUpload.files[0];
  
      // Memeriksa apakah file telah dipilih
      if (!file) {
        predictionContainer.innerText = 'Please select an image file.';
        return;
      }
  
      try {
        // Memprediksi gambar
        const predictedClassIndex = await predictImage(model, file);
  
        // Menampilkan hasil prediksi
        let predictionText;
        if (predictedClassIndex === 0) {
          predictionText = 'Back Pod Rot';
        } else if (predictedClassIndex === 1) {
          predictionText = 'Healthy';
        } else if (predictedClassIndex === 2) {
          predictionText = 'Pod Borer';
        } else {
          predictionText = 'Unknown';
        }
  
        predictionContainer.innerText = `Predicted Class: ${predictionText}`;
      } catch (error) {
        predictionContainer.innerText = 'Failed to predict image. Please check the console for details.';
      }
    });
  }
  
  // Memanggil fungsi init untuk memulai aplikasi
  init();
  