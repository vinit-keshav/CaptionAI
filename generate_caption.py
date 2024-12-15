import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nltk
import argparse

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer


nltk.download('punkt')


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


WORKING_DIR = os.path.join(CURRENT_DIR, 'working_dir')


BEST_MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.keras')  # Updated to .keras
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')



def load_tokenizer(tokenizer_path):
    """Load the tokenizer from a pickle file."""
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
    return tokenizer

def load_trained_model(model_path):
    """Load the trained Keras model."""
    try:
        model = load_model(model_path)
        print("Trained model loaded successfully.")
        return model
    except ValueError as ve:
        print(f"Error loading model: {ve}")
        print("Ensure that the model was saved correctly and does not contain custom layers.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        exit(1)

def load_vgg16_model():
    """Load the VGG16 model for feature extraction."""
    vgg_model = VGG16()

    vgg_model = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    print("VGG16 model loaded for feature extraction.")
    return vgg_model

def idx_to_word(integer, tokenizer):
    """Convert an integer index to its corresponding word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_feature(image_path, vgg_model):
    """Extract features from an image using VGG16."""
    try:
        image = load_img(image_path, target_size=(224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def generate_caption(model, tokenizer, vgg_model, image_path, max_length):
    """Generate a caption for a given image."""
    feature = extract_feature(image_path, vgg_model)
    if feature is None:
        return "Error in extracting features from the image."

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq ', '').replace(' endseq', '')
    return final_caption

def display_image(image_path):
    """Display the image."""
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")


def main(image_path):
    """Main function to generate and display caption for an image."""

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Trained model not found at {BEST_MODEL_PATH}. Please train the model first.")
        return
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer not found at {TOKENIZER_PATH}. Please train the model first.")
        return


    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_trained_model(BEST_MODEL_PATH)
    vgg_model = load_vgg16_model()


    try:
        text_input_shape = model.input_shape[1]
        max_length = text_input_shape[1]
        print(f"Maximum Caption Length: {max_length}")
    except Exception as e:
        print(f"Error determining max_length from model's input shape: {e}")
        return


    caption = generate_caption(model, tokenizer, vgg_model, image_path, max_length)
    print("Generated Caption:", caption)


    display_image(image_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate captions for images using a trained model.')
    parser.add_argument('image', type=str, help='Path to the image file.')
    args = parser.parse_args()

    image_path = args.image

    if not os.path.isfile(image_path):
        print(f"Image file '{image_path}' does not exist.")
        exit(1)

    main(image_path)
