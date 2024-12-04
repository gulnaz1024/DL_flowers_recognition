import os
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('flower_model.pth', map_location=device)
model.eval()  # Set the model to evaluation mode

# Class names dictionary
class_names = {
    0: 'daisy',
    1: 'dandelion',
    2: 'roses',
    3: 'sunflowers',
    4: 'tulips'
}

# Image transformations (adjust as needed)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to home page (index page)
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        image = Image.open(filepath)
        image = transform(image).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            flower_class = predicted.item()

            # Get the probability for the predicted class
            probs = torch.nn.functional.softmax(output, dim=1)
            prob = probs[0][flower_class].item()

        # Return the predicted class, probability, and image path to the result page
        image_url = os.path.join('uploads', filename)  # Correct the image URL
        image_url = image_url.replace("\\", "/")  # Замените обратные слэши на косые

        return render_template('result.html', flower_class=class_names[flower_class], prob=prob, image_url=image_url)

    return 'File format not supported'


if __name__ == '__main__':
    app.run(debug=True)
