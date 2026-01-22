from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import os
from pathlib import Path
import io

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'model/monkey_classifier.pth'
LABELS_PATH = 'model/class_labels.json'
UPLOAD_FOLDER = 'static/uploads'
SAMPLES_FOLDER = 'static/samples'
IMG_SIZE = 224

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

# Global variables
model = None
class_labels = None
device = None

def create_model(num_classes):
    """Create the same model architecture as in training"""
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    return model

def load_model():
    """Load the trained model and class labels"""
    global model, class_labels, device
    
    try:
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        
        num_classes = len(class_labels)
        print(f"Loaded {num_classes} classes: {list(class_labels.values())}")
        
        # Create model
        model = create_model(num_classes)
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict_species(image):
    """Predict monkey species from image"""
    try:
        # Preprocess
        image_tensor = preprocess_image(image).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get species name
        predicted_idx = predicted_idx.item()
        confidence_score = confidence.item()
        species_name = class_labels[str(predicted_idx)]
        
        return species_name, confidence_score
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

def get_sample_image_path(species_name):
    """Get the path to the sample image for a species"""
    # Convert species name to filename (lowercase, replace spaces with underscores)
    filename = species_name.lower().replace(' ', '_') + '.jpg'
    print("Filename:", filename)
    sample_path = f"/static/samples/{filename}"
    
    # Check if file exists
    full_path = os.path.join(SAMPLES_FOLDER, filename)
    if not os.path.exists(full_path):
        # Try alternative extensions
        for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
            alt_filename = species_name.lower().replace(' ', '_') + ext
            alt_path = os.path.join(SAMPLES_FOLDER, alt_filename)
            if os.path.exists(alt_path):
                return f"/static/samples/{alt_filename}"
        
        print(f"Warning: Sample image not found for {species_name}")
        return None
    
    return sample_path

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/search', methods=['POST'])
def search_similar():
    """Main endpoint for image classification"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
        
        # Load image
        try:
            image = Image.open(file.stream)
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Predict species
        species_name, confidence = predict_species(image)
        
        # Get sample image path
        sample_path = get_sample_image_path(species_name)
        
        # Prepare response
        result = {
            'species': species_name,
            'similarity': float(confidence),  # Using confidence as similarity score
            'image_path': sample_path if sample_path else '/static/samples/placeholder.jpg'
        }
        
        response = {
            'results': [result],
            'prediction': {
                'species': species_name,
                'confidence': float(confidence)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in search_similar: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/static/samples/<path:filename>')
def serve_sample(filename):
    """Serve sample images"""
    return send_from_directory(SAMPLES_FOLDER, filename)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    print("="*60)
    print("MONKEY SPECIES CLASSIFIER API")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    load_model()
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)