import argparse
import torch
import torch.nn.functional as F
import ctc_utils
import cv2
import numpy as np

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.model)
model.eval()

# Read the dictionary
with open(args.voc_file, "r") as dict_file:
    dict_list = dict_file.read().splitlines()
int2word = {idx: word for idx, word in enumerate(dict_list)}

# Process the image
image = cv2.imread(args.image, False)
image = ctc_utils.resize(
    image, model.input_height
)  # Assuming `model.input_height` is set appropriately
image = ctc_utils.normalize(image)
image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
image = torch.from_numpy(image).to(device).float()

# Forward pass through the model
with torch.no_grad():
    logits = model(image)
    seq_len = torch.full(
        (logits.size(0),), logits.size(1), device=device, dtype=torch.long
    )
    decoded = F.ctc_greedy_decode(logits, seq_len)

# Convert predictions to strings
str_predictions = ctc_utils.sparse_tensor_to_strs(decoded)
for w in str_predictions[0]:
    print(int2word[w], end="\t")
