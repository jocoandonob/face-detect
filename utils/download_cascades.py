import os
import urllib.request
import cv2

def download_cascade_files():
    """Download cascade classifier XML files from OpenCV GitHub repository."""
    # Create data directory if it doesn't exist
    data_dir = os.path.join('models', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Define files to download
    cascade_files = {
        'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
        'haarcascade_smile.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml'
    }
    
    # Download each file
    for filename, url in cascade_files.items():
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded to {file_path}")
        else:
            print(f"File {filename} already exists at {file_path}")

if __name__ == "__main__":
    download_cascade_files()
    print("Cascade files download complete.") 