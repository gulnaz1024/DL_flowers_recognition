import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image

# Initialize Flask app
app = Flask(__name__)  # Use __name__ instead of name

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('flower_model.pth', map_location=device)
model.eval()  # Set the model to evaluation mode

# Image transformations (you can adjust these based on your dataset)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Route to home page
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

    # Process the image
    image = Image.open(file)
    image = transform(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        # Assuming you have a function to map output to flower class (index)
        _, predicted = torch.max(output, 1)
        flower_class = predicted.item()

    return f'Predicted flower class: {flower_class}'

# Correct entry point check
if __name__ == '__main__':  # Use __name__ instead of name
    app.run(debug=True)
