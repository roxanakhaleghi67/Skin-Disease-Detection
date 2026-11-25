# skin_diagnosis_app/diagnose.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

class SkinAnalyzer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        pred = self.model.predict(x)
        return pred.argmax(), pred.max()

# skin_diagnosis_app/README.md
"""
# Skin Disease Classifier

Image classifier for common skin conditions. Requires trained CNN model.

## Example:
```python
analyzer = SkinAnalyzer("skin_model.h5")
id, conf = analyzer.predict("rash.jpg")
```
"""