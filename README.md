# Monkey Species Image Classifier

A web application that classifies monkey images into 10 species categories using Flask and TensorFlow.

## Prerequisites

- Python 3.8+
- Node.js (optional, for frontend dev, but this project uses CDN)

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download and Prepare Data**
    This script downloads the "10-monkey-species" dataset from Kaggle and organizes it.
    ```bash
    python download_data.py
    ```

3.  **Train the Model**
    Trains a MobileNetV2 model on the dataset. This will take some time (approx 10-15 epochs).
    It also generates `model/class_labels.json`.
    ```bash
    python train_model.py
    ```

4.  **Prepare Sample Images**
    Extracts representative images for the UI results.
    ```bash
    python prepare_samples.py
    ```

5.  **Run the Server**
    ```bash
    python app.py
    ```

## Usage

1.  Open your browser and navigate to `http://localhost:5000`.
2.  Upload an image of a monkey.
3.  The application will identify the species and show a sample image of that species.

## API Endpoints

-   **POST /api/search**
    -   Input: `multipart/form-data` with `image` file.
    -   Output: JSON with predicted species and similarity score.
-   **GET /health**
    -   Status check.

## Directory Structure

-   `app.py`: Flask backend.
-   `train_model.py`: Model training script.
-   `download_data.py`: Dataset downloader.
-   `prepare_samples.py`: Sample image generator.
-   `model/`: Stores trained model `.h5` and labels.
-   `static/`: CSS, JS, uploads, and sample images.
-   `templates/`: HTML files.
-   `training/`: Dataset directory.
