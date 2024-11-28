import argparse
import torch
import ctc_utils
import cv2
import numpy as np
from ctc_model import CTC_CRNN, default_model_params

# Argument parser
parser = argparse.ArgumentParser(
    description="Decode a music score image with a trained model (CTC)."
)
parser.add_argument(
    "-image", dest="image", type=str, required=True, help="Path to the input image."
)
parser.add_argument(
    "-model", dest="model", type=str, required=True, help="Path to the trained model."
)
parser.add_argument(
    "-vocabulary",
    dest="voc_file",
    type=str,
    required=True,
    help="Path to the vocabulary file.",
)
args = parser.parse_args()

# Load vocabulary
with open(args.voc_file, "r") as dict_file:
    dict_list = dict_file.read().splitlines()
    int2word = {idx: word for idx, word in enumerate(dict_list)}

# Load model configuration
img_height = 128  # Ensure this matches the model training configuration
vocabulary_size = len(int2word)
params = default_model_params(img_height, vocabulary_size)

# Load trained model
device = torch.device("cpu")# torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTC_CRNN(params)
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()

# Process input image
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
image = ctc_utils.resize(image, img_height)  # Resize to the model's height
image = ctc_utils.normalize(image)  # Normalize image values
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = np.expand_dims(image, axis=0)  # Add channel dimension (grayscale)

# Convert image to PyTorch tensor and move to device
image_tensor = torch.tensor(image, dtype=torch.float32).to(
    device
)  # Shape: [1, 1, height, width]

# Sequence lengths (based on width reduction during model creation)
seq_lengths = torch.tensor(
    [image_tensor.shape[3] // params["conv_pooling_size"][0][1]], dtype=torch.int32
).to(device)

# Perform prediction
with torch.no_grad():
    logits, _ = model(image_tensor, seq_lengths)
    log_probs = logits.log_softmax(2)  # Apply log softmax
    predictions = torch.argmax(log_probs, dim=2).squeeze(1)  # Greedy decoding

# Convert predictions to text
str_predictions = ctc_utils.decode_predictions(
    predictions.cpu().numpy()
)  # Assuming this utility exists
for idx in str_predictions[0]:
    print(int2word[idx], end="\t")
