# Handwritten Multi-Digit Recognition

This project implements a deep learning model for recognizing handwritten multi-digit numbers using TensorFlow and Keras. The model is trained on the MNIST dataset and can be used to predict sequences of digits from hand-drawn images.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Accuracy](#model-training-and-accuracy)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Handwritten digit recognition is a classic problem in the field of computer vision and machine learning. This project extends the basic MNIST digit recognition to recognize sequences of digits (multi-digit numbers) from handwritten images.

## Project Structure

```
.
├── recognize_digits.py
├── mnist_model.h5
├── [other source files]
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rithika-Emmadi/Hand-multi-didgit-recognition.git
   cd Hand-multi-didgit-recognition
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model and recognize digits, run:

```bash
python recognize_digits.py
```

The script will output the recognized number from the test image and save the trained model to `mnist_model.h5`.

## Model Training and Accuracy

The model was trained for 5 epochs using the MNIST dataset. Below are the training and validation accuracies and losses for each epoch:

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|------------------|---------------|---------------------|-----------------|
| 1     | 0.8353           | 0.5408        | 0.9810              | 0.0789          |
| 2     | 0.9619           | 0.1565        | 0.9846              | 0.0635          |
| 3     | 0.9699           | 0.1392        | 0.9867              | 0.0668          |
| 4     | 0.9693           | 0.1794        | 0.9885              | 0.0628          |
| 5     | 0.9705           | 0.2055        | 0.9876              | 0.0817          |

- **Final Training Accuracy:** 0.9705
- **Final Validation Accuracy:** 0.9876

## Results

After training, the model successfully recognizes handwritten multi-digit numbers. Example output from the script:

```
Recognized Number: 123
```

## Saving and Loading the Model

- The trained model is saved as `mnist_model.h5` (HDF5 format).
- To load and use the model for inference:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('mnist_model.h5')
  ```

> **Note:** The HDF5 format is considered legacy. For new projects, consider saving models using the native Keras format:
> ```python
> model.save('my_model.keras')
> ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License

This project is licensed under the MIT License.
