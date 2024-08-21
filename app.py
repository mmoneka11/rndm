from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
custom_model = load_model("cifar10_model.h5")

# CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array.astype('float32') / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# HTML template for file upload and prediction result
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Image Classification</title>
</head>
<body>
    <h1>Upload an image for CIFAR-10 classification</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    {% if predicted_class %}
    <h2>Prediction Result</h2>
    <p>Predicted Class: {{ predicted_class }}</p>
    <p>Confidence: {{ predicted_confidence }}%</p>
    <img src="{{ url_for('static', filename=image_path) }}" alt="Uploaded Image" />
    {% endif %}
</body>
</html>
"""

# Route to handle file upload and prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join('/content/', file.filename)
            file.save(filepath)

            # Preprocess the image
            img_array = preprocess_image(filepath)

            # Make prediction
            predictions = custom_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]
            predicted_confidence = (predictions[0][predicted_class_index]) * 100

            # Render the HTML template with prediction results
            return render_template_string(HTML_TEMPLATE,
                                           predicted_class=predicted_class,
                                           predicted_confidence=f'{predicted_confidence:.2f}'
                                           )
    return render_template_string(HTML_TEMPLATE)

# Run the app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port='5000')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 80))  # Use PORT environment variable or default to 80
    app.run(debug=False, port=port)
