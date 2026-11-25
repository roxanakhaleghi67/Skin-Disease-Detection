"""Example usage for the skin disease classifier.

Run:  python main.py path/to/skin_model.h5 path/to/image.jpg
"""

import sys

from skin_diagnosis_app.diagnose import SkinAnalyzer


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python main.py <model.h5> <image>")
    model_path, image_path = sys.argv[1], sys.argv[2]
    analyzer = SkinAnalyzer(model_path)
    class_id, confidence = analyzer.predict(image_path)
    print(f"class index: {class_id} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
