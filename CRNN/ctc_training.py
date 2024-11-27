import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from ctc_model import CTC_CRNN, default_model_params
from primus import CTC_PriMuS  # Assuming similar data handling as TensorFlow script
import ctc_utils  # Utility methods (unchanged)

parser = argparse.ArgumentParser(description="Train model.")
parser.add_argument(
    "-corpus", dest="corpus", type=str, required=True, help="Path to the corpus."
)
parser.add_argument(
    "-set", dest="set", type=str, required=True, help="Path to the set file."
)
parser.add_argument(
    "-save_model",
    dest="save_model",
    type=str,
    required=True,
    help="Path to save the model.",
)
parser.add_argument(
    "-vocabulary",
    dest="voc",
    type=str,
    required=True,
    help="Path to the vocabulary file.",
)
parser.add_argument("-semantic", dest="semantic", action="store_true", default=False)
args = parser.parse_args()

# Load dataset
primus = CTC_PriMuS(args.corpus, args.set, args.voc, args.semantic, val_split=0.1)

# Model parameters
img_height = 128
params = default_model_params(img_height, primus.vocabulary_size)
model = CTC_CRNN(params)

# Training configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ctc_loss = nn.CTCLoss(blank=params["vocabulary_size"])
optimizer = optim.Adam(model.parameters())

# Training loop
max_epochs = 64000
dropout = 0.5
batch_size = params["batch_size"]

for epoch in range(max_epochs):
    model.train()
    batch = primus.nextBatch(params)

    inputs = torch.tensor(batch["inputs"], dtype=torch.float32).to(device)
    seq_lengths = torch.tensor(batch["seq_lengths"], dtype=torch.int32).to(device)
    targets = ctc_utils.sparse_tuple_from(batch["targets"])

    # Forward pass
    logits, seq_lengths_out = model(inputs, seq_lengths)

    log_probs = logits.log_softmax(2)
    loss = ctc_loss(log_probs, targets, seq_lengths_out, targets.shape[0])

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        # Validation
        print(f"Loss value at epoch {epoch}: {loss.item()}")
        print("Validating...")

        model.eval()
        with torch.no_grad():
            validation_batch, validation_size = primus.getValidation(params)
            val_idx = 0
            val_ed = 0
            val_len = 0
            val_count = 0

            while val_idx < validation_size:
                mini_batch_inputs = torch.tensor(
                    validation_batch["inputs"][val_idx : val_idx + batch_size],
                    dtype=torch.float32,
                ).to(device)
                mini_batch_seq_lengths = torch.tensor(
                    validation_batch["seq_lengths"][val_idx : val_idx + batch_size],
                    dtype=torch.int32,
                ).to(device)

                logits, seq_lengths_out = model(
                    mini_batch_inputs, mini_batch_seq_lengths
                )

                predictions = torch.argmax(logits, dim=2)
                str_predictions = ctc_utils.decode_predictions(predictions)

                for i, pred in enumerate(str_predictions):
                    ed = ctc_utils.edit_distance(
                        pred, validation_batch["targets"][val_idx + i]
                    )
                    val_ed += ed
                    val_len += len(validation_batch["targets"][val_idx + i])
                    val_count += 1

                val_idx += batch_size

            print(
                f"[Epoch {epoch}] SER: {100.0 * val_ed / val_len:.2f}% "
                f"({val_ed / val_count:.2f} average edit distance)."
            )

        # Save model
        torch.save(model.state_dict(), f"{args.save_model}_epoch_{epoch}.pth")
        print("Model saved.")
