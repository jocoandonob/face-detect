# Face Insight AI

A facial recognition and analysis application built with Python.

## Features

- Facial recognition and detection
- User authentication and management
- Web interface using Flask/FastAPI

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/FaceInsightAI.git
cd FaceInsightAI
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the required cascade files for facial detection
```bash
python utils/download_cascades.py
```

## Usage

To run the application:

```bash
python app.py
```

## Project Structure

```
FaceInsightAI/
├── app/            # Application code
├── models/         # ML models and detection code
│   └── data/       # Cascade classifier XML files
├── static/         # Static assets (CSS, JS, images)
├── templates/      # HTML templates
├── tests/          # Test files
├── utils/          # Utility scripts
├── .gitignore      # Git ignore file
├── pyproject.toml  # Project configuration
├── README.md       # This file
└── requirements.txt # Python dependencies
```

## License

[MIT](LICENSE) 