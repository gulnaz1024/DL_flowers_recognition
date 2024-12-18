import os
import shutil
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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('flower_model.pth', map_location=device)
model.eval()  # Set the model to evaluation mode

# Class names dictionary
class_names = {
    0: 'Daisy',
    1: 'Dandelion',
    2: 'Roses',
    3: 'Sunflowers',
    4: 'Tulips'
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


# Function to clean up the uploads folder when the number of files exceeds 20
def clean_up_uploads_folder(max_files=25):
    # List all files in the uploads folder
    files = os.listdir(UPLOAD_FOLDER)

    # Check if the number of files exceeds max_files
    if len(files) > max_files:
        # Empty the uploads folder by deleting all files
        for file in files:
            if file != ".gitkeep":  # Skip the .gitkeep file
                file_path = os.path.join(UPLOAD_FOLDER, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Delete the file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directories if any
                except Exception as e:
                    print(f"Error while deleting file {file_path}: {e}")

# Route to home page (index page)
@app.route('/', methods=['GET', 'POST'])
def home():
    flower_class = None
    prob = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            # Clean up the uploads folder if the number of files exceeds 20
            clean_up_uploads_folder()

            # Save the new file
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
                flower_class = class_names[predicted.item()]

                # Get the probability for the predicted class
                probs = torch.nn.functional.softmax(output, dim=1)
                prob = probs[0][predicted.item()].item()

            # Path to the uploaded image
            image_url = os.path.join('uploads', filename).replace("\\", "/")  # Ensure forward slashes

    return render_template('index.html', flower_class=flower_class, prob=prob, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
