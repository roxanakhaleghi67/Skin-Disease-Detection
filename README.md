# Skin Disease Classifier

Image classifier for common skin conditions using a trained convolutional neural network (Keras/TensorFlow).

## Features

- Loads a trained Keras model (`.h5`) from disk
- Resizes input images to 224×224 and normalizes pixel values
- Runs a forward pass and returns the predicted class index and confidence
- Simple, class-based API (`SkinAnalyzer`)

## Requirements

- Python 3.8 or newer
- `tensorflow` (2.x)
- `numpy`
- `pillow` (Pillow), used by the Keras image utilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The repository contains a single source module that defines `SkinAnalyzer`. Import it (here as `diagnose`) and call it with a model path and image path:

```python
from diagnose import SkinAnalyzer

analyzer = SkinAnalyzer("skin_model.h5")
class_id, confidence = analyzer.predict("rash.jpg")
print(class_id, confidence)
```

> **Note:** A trained model file is **not** included in this repository. You must supply your own `SkinAnalyzer`-compatible model before predictions can be made. The class index maps to whatever label order your model was trained with.

> **Disclaimer:** This project is provided for educational and research purposes only. It is **not** a medical device and must not be used to diagnose, treat, or rule out any condition. Always consult a qualified healthcare professional for medical advice.

## Project Structure

```text
Skin-Disease-Detection/
├── Skin Disease Detection.py   # SkinAnalyzer implementation
├── requirements.txt
└── README.md
```

## License

This project is released under the **MIT License**.

## Author

**Roxana Khaleghi**
