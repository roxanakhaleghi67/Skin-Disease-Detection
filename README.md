# Skin Disease Image Classifier

Classifies a photo of a skin condition with a trained convolutional neural network
(TensorFlow/Keras).

## What it does

`skin_disease_detection.py` defines `SkinAnalyzer`: it resizes the image to 224×224,
scales pixel values to `[0, 1]`, runs the model and returns the predicted class index
and its confidence.

```python
from skin_disease_detection import SkinAnalyzer

analyzer = SkinAnalyzer("skin_model.h5")
class_id, confidence = analyzer.predict("rash.jpg")
```

## Requirements

- Python 3.9+
- `tensorflow`, `numpy` (see `requirements.txt`)

```bash
python -m pip install -r requirements.txt
```

## Notes and limitations

- A trained model file is **not** included; supply one trained on your own dataset.
- Class indices are unlabelled here; keep the training label order alongside the model.
- **Not a medical device.** Output is a research signal only and must not be used for
  diagnosis or treatment decisions.

## Status

Single-module prototype. Planned work: label mapping, a training notebook, and a small demo app.

## License

No license file is present yet. Until one is added, the default copyright applies.
