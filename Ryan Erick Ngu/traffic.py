import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    # Remove one-hot encoding of labels
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # Check if the data directory exists
    abs_path = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        print(f"Debug: Provided data directory path (absolute): {abs_path}")
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist (absolute path: '{abs_path}'). "
            "Please provide the correct path to the data directory."
        )
    
    images = []
    labels = []
    
    # Iterate through each category folder (0 to NUM_CATEGORIES-1)
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))
        
        # Check if category directory exists
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory {category_dir} does not exist. Skipping this category.")
            continue
        
        # Iterate through images in the category directory
        for filename in os.listdir(category_dir):
            file_path = os.path.join(category_dir, filename)
            
            # Read image using OpenCV
            image = cv2.imread(file_path)
            
            # Skip non-image files or invalid images
            if image is None:
                print(f"Warning: Failed to load image {file_path}. Skipping.")
                continue
            
            # Resize image to the specified dimensions
            image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            
            # Append resized image and its label
            images.append(image_resized)
            labels.append(label)
    
    # Check if any data was loaded
    if len(images) == 0 or len(labels) == 0:
        raise ValueError(
            "No data loaded. Ensure the data directory is correctly structured and contains valid images. "
            f"Checked directory: {data_dir}"
        )
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize pixel values to be between 0 and 1
    images = images.astype('float32') / 255.0
    
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.Sequential()

    # Add convolutional layers, pooling layers, and dropout layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))  # Dropout layer for regularization
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Flatten the output of the convolutional layers
    model.add(tf.keras.layers.Flatten())
    
    # Add dense (fully connected) layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))  # Output layer
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    main()
