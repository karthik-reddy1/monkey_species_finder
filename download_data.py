import kagglehub
import shutil
import os

def setup_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("slothkong/10-monkey-species")
    print("Dataset downloaded to:", path)

    target_dir = os.path.join(os.getcwd(), 'training')
    
    # Destination path for the inner dataset folder
    # The dataset usually contains 'training' and 'validation' folders inside
    # We want project/training/10-monkey-species/{training, validation}
    destination = os.path.join(target_dir, '10-monkey-species')
    
    if os.path.exists(destination):
        print(f"Destination {destination} already exists. Cleaning up...")
        shutil.rmtree(destination)
    
    print(f"Copying files to {destination}...")
    shutil.copytree(path, destination)
    print("Dataset setup complete.")

if __name__ == "__main__":
    setup_dataset()
