import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU to avoid potential crashes

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

MODEL_PATH = "mnist_model.h5"

# Create a new MNIST CNN model
def create_mnist_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load model or train if not available
if os.path.exists(MODEL_PATH):
    mnist_model = load_model(MODEL_PATH)
    print("✅ Loaded saved MNIST model.")
else:
    print("🚀 Training new MNIST model...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    mnist_model = create_mnist_model()
    mnist_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    mnist_model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

# Extract digits from thresholded image
def extract_digits(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            digit = thresh_img[y:y + h, x:x + w]
            square_size = max(h, w)
            padded_digit = np.zeros((square_size, square_size), dtype=np.uint8)
            y_offset = (square_size - h) // 2
            x_offset = (square_size - w) // 2
            padded_digit[y_offset:y_offset + h, x_offset:x_offset + w] = digit
            digit = cv2.resize(padded_digit, (28, 28))
            digit = digit.astype('float32') / 255.0
            digit_images.append((x, digit))

    digit_images = sorted(digit_images, key=lambda x: x[0])  # Sort left to right
    return [img for _, img in digit_images]

# Predict digit sequence
def recognize_digits(digit_images):
    digits = []
    for digit in digit_images:
        digit = np.expand_dims(digit, axis=(0, -1))  # Shape: (1, 28, 28, 1)
        prediction = np.argmax(mnist_model.predict(digit, verbose=0), axis=-1)[0]
        digits.append(str(prediction))
    return "".join(digits)

# Main recognition function
def recognize_number(image_path):
    thresh_img = preprocess_image(image_path)
    digit_images = extract_digits(thresh_img)
    if not digit_images:
        return "No digits found"
    return recognize_digits(digit_images)

# Example usage — change the path as needed
image_path = "/Users/rithikaemmadi/Downloads/image.png"
print("Recognized Number:", recognize_number(image_path))