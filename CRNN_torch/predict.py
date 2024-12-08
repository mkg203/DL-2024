import torch
import torch.nn as nn
import argparse
from PIL import Image
import numpy as np
from ctc_model import CTC_CRNN, default_model_params
import os

def load_and_preprocess_image(image_path, target_height=128):
    """Load and preprocess an image for the model."""
    # Load image in grayscale
    image = Image.open(image_path).convert('L')
    
    # Calculate new width to maintain aspect ratio
    aspect_ratio = image.size[0] / image.size[1]
    new_width = int(target_height * aspect_ratio)
    
    # Resize image
    image = image.resize((new_width, target_height))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add channel dimension and batch dimension
    img_array = img_array[np.newaxis, np.newaxis, :, :]
    
    return img_array, new_width

def load_vocabulary(vocab_path):
    """Load vocabulary from file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = [line.strip() for line in f]
    return vocabulary

def decode_prediction(prediction, vocabulary):
    """Decode the model's prediction using the vocabulary."""
    decoded = []
    previous = None
    for token in prediction:
        if token != previous and token < len(vocabulary):  # Skip CTC blank token
            decoded.append(vocabulary[token])
        previous = token
    return ' '.join(decoded)

def main():
    parser = argparse.ArgumentParser(description="Predict using trained CTC-CRNN model.")
    parser.add_argument("-model", required=True, help="Path to the saved model checkpoint")
    parser.add_argument("-image", required=True, help="Path to the input image")
    parser.add_argument("-vocabulary", required=True, help="Path to the vocabulary file")
    args = parser.parse_args()

    # Load vocabulary
    vocabulary = load_vocabulary(args.vocabulary)
    
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = default_model_params(img_height=128, vocabulary_size=len(vocabulary))
    model = CTC_CRNN(params)
    
    # Load model weights
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Process image
    img_array, width = load_and_preprocess_image(args.image)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).to(device)
    seq_lengths = torch.tensor([width // 8], dtype=torch.int32).to(device)  # Assuming 3 pooling layers with size 2

    # Make prediction
    with torch.no_grad():
        logits, output_lengths = model(img_tensor, seq_lengths)
        predictions = torch.argmax(logits.log_softmax(2), dim=2).transpose(0, 1)
        
        # Decode predictions
        print(predictions)
        for pred_seq in predictions:
            decoded_text = decode_prediction(pred_seq.cpu().numpy(), vocabulary)
            print(f"\nPredicted text: {decoded_text}")

if __name__ == "__main__":
    main()
