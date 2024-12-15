import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import re
import nltk
import csv

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# Ensure NLTK data is downloaded
nltk.download('punkt')

# =========================
# Configuration
# =========================

# Get the current script directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the Flickr dataset directory
BASE_DIR = os.path.join(CURRENT_DIR, 'flickr')

# Path to store intermediate files and models
WORKING_DIR = os.path.join(CURRENT_DIR, 'working_dir')

# Create working directory if it doesn't exist
os.makedirs(WORKING_DIR, exist_ok=True)

# =========================
# Path Verification
# =========================

# Check if the Images directory exists
images_directory = os.path.join(BASE_DIR, 'Images')
if not os.path.isdir(images_directory):
    raise FileNotFoundError(f"Images directory not found in {images_directory}")

# Check if the captions file exists
captions_file = os.path.join(BASE_DIR, 'captions.txt')
if not os.path.isfile(captions_file):
    raise FileNotFoundError(f"Captions file not found at {captions_file}")

print("Directory paths are correctly set up.")

# =========================
# Feature Extraction
# =========================

def extract_features(directory, model):
    """Extract features from each image using the VGG16 model."""
    features = {}
    for img_name in tqdm(os.listdir(directory), desc="Extracting Features"):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, img_name)
            try:
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                feature = model.predict(image, verbose=0)
                image_id = os.path.splitext(img_name)[0]  # Remove file extension
                features[image_id] = feature
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    return features

# Load VGG16 model + higher level layers
vgg_model = VGG16()
# Remove the last layer (classification layer) to get image features
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
print("VGG16 Model Summary:")
vgg_model.summary()

# Extract features and save to a file
image_dir = images_directory  # Ensure this path is correct
features_path = os.path.join(WORKING_DIR, 'features.pkl')

if not os.path.exists(features_path):
    print("Starting feature extraction...")
    features = extract_features(image_dir, vgg_model)
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Extracted features saved to {features_path}")
else:
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    print(f"Loaded features from {features_path}")

# =========================
# Load and Preprocess Captions
# =========================

def load_captions(captions_file):
    """Load captions from the captions file."""
    mapping = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        for row in tqdm(reader, desc="Processing Captions"):
            if len(row) != 2:
                print(f"Skipping malformed line: {row}")
                continue
            image_name, caption = row
            image_id = os.path.splitext(image_name)[0]  # Remove file extension
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
    return mapping

# Load captions
mapping = load_captions(captions_file)
print(f"Total Images with Captions: {len(mapping)}")

def clean_caption(caption):
    """Clean a single caption."""
    caption = caption.lower()
    caption = re.sub(r'[^a-zA-Z\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip()
    caption = 'startseq ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' endseq'
    return caption

def clean_mapping(mapping):
    """Clean all captions in the mapping."""
    for key, captions in mapping.items():
        for i in range(len(captions)):
            captions[i] = clean_caption(captions[i])

# Before cleaning
if len(mapping) > 0:
    sample_image_id = list(mapping.keys())[0]
    print("Before Cleaning:", mapping[sample_image_id])
else:
    print("No captions to clean.")

# Clean captions
clean_mapping(mapping)

# After cleaning
if len(mapping) > 0:
    print("After Cleaning:", mapping[sample_image_id])
else:
    print("No captions were cleaned.")

# Save cleaned mapping
clean_mapping_path = os.path.join(WORKING_DIR, 'clean_mapping.pkl')
with open(clean_mapping_path, 'wb') as f:
    pickle.dump(mapping, f)
print(f"Cleaned captions saved to {clean_mapping_path}")

# Prepare all captions
all_captions = []
for key in mapping:
    all_captions.extend(mapping[key])
print(f"Total Captions: {len(all_captions)}")
if len(all_captions) > 0:
    print("Sample Captions:", all_captions[:10])
else:
    print("No captions available.")

# =========================
# Tokenization
# =========================

# Initialize and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {vocab_size}")

# Determine the maximum sequence length
max_length = max(len(caption.split()) for caption in all_captions)
print(f"Maximum Caption Length: {max_length}")

# Save tokenizer
tokenizer_path = os.path.join(WORKING_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {tokenizer_path}")

# =========================
# Train-Test Split
# =========================

image_ids = list(mapping.keys())
np.random.seed(42)
np.random.shuffle(image_ids)
split = int(len(image_ids) * 0.90)
train_ids = image_ids[:split]
test_ids = image_ids[split:]
print(f"Training Samples: {len(train_ids)}, Testing Samples: {len(test_ids)}")

# =========================
# Data Generator
# =========================

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """Generate data for the model."""
    X1, X2, y = [], [], []
    while True:
        for key in data_keys:
            captions = mapping[key]
            feature = features.get(key)
            if feature is None:
                continue
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature[0])
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)  # Changed to tuple
                        X1, X2, y = [], [], []

# =========================
# Define the Model
# =========================

# Image feature extractor model
inputs1 = Input(shape=(4096,), name='image_input')
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Sequence processor model
inputs2 = Input(shape=(max_length,), name='text_input')
se1 = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Combined model
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Image Captioning Model Summary:")
model.summary()

# Optional: Visualize the model architecture
# plot_model(model, to_file=os.path.join(WORKING_DIR, 'model.png'), show_shapes=True, show_layer_names=True)

# =========================
# Training the Model
# =========================

# Training parameters
epochs = 20
batch_size = 32
steps = len(train_ids) // batch_size

# Start training
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    generator = data_generator(train_ids, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # Save the model after each epoch in .keras format
    model.save(os.path.join(WORKING_DIR, f'model_epoch_{epoch+1}.keras'))
    print(f"Model saved for epoch {epoch+1}")

# Save the final model in .keras format
best_model_path = os.path.join(WORKING_DIR, 'best_model.keras')
model.save(best_model_path)
print(f"Trained model saved to {best_model_path}")

# =========================
# Helper Functions for Inference
# =========================

def idx_to_word(integer, tokenizer):
    """Convert an integer index to its corresponding word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    """Generate a caption for a given image."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# =========================
# Evaluation with BLEU Scores
# =========================

def evaluate_model(model, features, mapping, tokenizer, max_length, test_keys):
    """Evaluate the model using BLEU scores."""
    from nltk.translate.bleu_score import corpus_bleu  # Ensure it's imported here

    actual, predicted = list(), list()
    for key in tqdm(test_keys, desc="Evaluating"):
        captions = mapping[key]
        feature = features.get(key)
        if feature is None:
            continue
        y_pred = predict_caption(model, feature, tokenizer, max_length)
        actual_captions = [nltk.word_tokenize(caption) for caption in captions]  # Ensure no language specified
        y_pred = nltk.word_tokenize(y_pred)  # Ensure no language specified
        predicted.append(y_pred)
        actual.append(actual_captions)

    # Calculate BLEU scores
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

# =========================
# Load the Best Model
# =========================

# Note: Since we've just trained and saved the model, we don't need to load it again here.
# Instead, evaluation should be handled separately or after training without reloading.

# =========================
# Evaluate the Model
# =========================

if len(test_ids) > 0:
    evaluate_model(model, features, mapping, tokenizer, max_length, test_ids)
else:
    print("No test data available for evaluation.")

# =========================
# Visualize the Results
# =========================

def generate_caption_for_image(model, image_name, features, tokenizer, max_length, base_dir=BASE_DIR):
    """Generate and display caption for a given image."""
    image_id = os.path.splitext(image_name)[0]
    img_path = os.path.join(base_dir, 'Images', image_name)
    try:
        image = Image.open(img_path)
    except Exception as e:
        print(f"Error opening image {image_name}: {e}")
        return
    captions = mapping.get(image_id, [])

    print('---------------------Actual Captions---------------------')
    for caption in captions:
        print(caption)

    feature = features.get(image_id)
    if feature is None:
        print("No features found for this image.")
        return

    y_pred = predict_caption(model, feature, tokenizer, max_length)
    print('--------------------Predicted Caption--------------------')
    print(y_pred)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage:
# Replace 'image1.jpg' with an actual image name from your dataset
if len(mapping) > 0:
    sample_image_id = list(mapping.keys())[0]
    sample_image_name = sample_image_id + ".jpg"  # Assuming images have .jpg extension
    generate_caption_for_image(model, sample_image_name, features, tokenizer, max_length)
else:
    print("No images available to generate captions.")

# =========================
# Testing with a New Image
# =========================

def extract_feature_from_image(image_path, model):
    """Extract feature from a single image using VGG16."""
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        return feature
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def test_with_real_image(model, image_path, tokenizer, max_length, vgg_model, working_dir=WORKING_DIR):
    """Generate caption for a new image not in the dataset."""
    # Extract features
    feature = extract_feature_from_image(image_path, vgg_model)
    if feature is None:
        print("Failed to extract features from the image.")
        return
    # Predict caption
    caption = predict_caption(model, feature, tokenizer, max_length)
    # Display the image and caption
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
    print("Generated Caption:", caption)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

