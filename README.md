# Emotion Detection App

An advanced emotion detection application built with Python, Tkinter, and OpenCV. This app detects emotions in real-time using your webcam or from uploaded images.

## Features

- **Real-time Emotion Detection**: Detects emotions like Happy, Sad, Angry, Surprise, Neutral, Fear, and Disgust via webcam.
- **Image Scanning**: Upload an image to analyze and detect emotions in faces.
- **Visual Trends**: Real-time graph showing emotion trends.
- **Filters**: Apply various visual filters (Sepia, Cartoon, Blur, etc.) to the camera feed.
- **Save & Export**: Save snapshots with emotion labels and export session data to CSV.

## Requirements

- Python 3.7+
- A webcam (for real-time detection)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Amlan-109/emotion-detection-app.git
   cd emotion-detection-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python emotion_detection_app.py
```

### Note on Login
The application currently has a simple login screen.
- **Username**: Amlan
- **Password**: amlan123

(You can modify the `login()` function in `emotion_detection_app.py` to change or remove this).

## Technologies Used

- **OpenCV**: For image processing and video capture.
- **FER (Facial Expression Recognition)**: For emotion detection.
- **Tkinter**: For the Graphical User Interface (GUI).
- **Pillow**: For image handling in the GUI.

## License

[MIT License](LICENSE)