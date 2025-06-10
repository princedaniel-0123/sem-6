# Sign Language to Text and Speech Conversion

This project is a Sign Language Recognition system that uses computer vision and deep learning to convert hand gestures (sign language) into text and speech. It also features a chatbot interface powered by the Groq API for interactive conversations.

## Features

- Real-time hand gesture recognition using webcam
- Deep learning model for sign classification ([cnn8grps_rad1_model.h5](cnn8grps_rad1_model.h5))
- Tkinter-based GUI for displaying predictions and chat interface
- Text-to-speech output using `pyttsx3`
- Chatbot integration via Groq API
- Word suggestions using `pyenchant`
- Data collection scripts for building custom datasets

## Directory Structure

```
.
├── .env
├── cnn.py
├── cnn8grps_rad1_model.h5
├── data_collection_binary.py
├── data_collection_final.py
├── final_pred.py
├── landmark.py
├── prediction_wo_gui.py
├── white.jpg
├── AtoZ_3.1/
│   ├── A/
│   ├── B/
│   └── ... (folders for each letter)
```

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Keras
- cvzone
- pyttsx3
- pyenchant
- python-dotenv
- Pillow
- groq (Groq API Python client)
- tkinter (usually included with Python)
- [Download the pre-trained model](cnn8grps_rad1_model.h5) and place it in the project root.

Install dependencies with:

```sh
pip install opencv-python numpy keras cvzone pyttsx3 pyenchant python-dotenv pillow groq
```

## Setup

1. **Clone the repository** and navigate to the project directory.
2. **Create a `.env` file** in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```
3. **Ensure `white.jpg`** is present in the root directory. If not, you can generate it using:
    ```python
    import numpy as np
    import cv2
    white = np.ones((400,400,3), np.uint8) * 255
    cv2.imwrite("white.jpg", white)
    ```
4. **Prepare the dataset** in the `AtoZ_3.1/` directory if you want to retrain or collect new data.

## Usage

### Run the Main Application

```sh
python final_pred.py
```

- The GUI will open, showing the webcam feed and prediction panels.
- Make hand signs in front of the camera to see predictions.
- The recognized text will appear in the sentence panel.
- Use the chat input or let the system auto-send recognized sentences to the chatbot.
- Use the "Speak" button to hear the recognized sentence.

### Data Collection

- Use `data_collection_binary.py` or `data_collection_final.py` to collect new gesture images for training.

### Model Training

- Use `cnn.py` to train or fine-tune the model on your dataset.

### Landmark Extraction

- Use `landmark.py` to extract and save hand/pose/face landmarks from videos.

## Notes

- The project uses the [cvzone](https://github.com/cvzone/cvzone) HandTrackingModule for hand detection.
- The chatbot requires a valid Groq API key.
- Word suggestions are powered by [pyenchant](https://pyenchant.github.io/pyenchant/).

## Acknowledgements

- [cvzone](https://github.com/cvzone/cvzone)
- [Groq API](https://console.groq.com/)
- [MediaPipe](https://mediapipe.dev/)
- [Keras](https://keras.io/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

## License

MIT License

---

**Contributions are welcome!**
