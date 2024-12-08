import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from ctc_model import CTC_CRNN, default_model_params
from primus import CTC_PriMuS
import ctc_utils    

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
primus = CTC_PriMuS(args.set, args.corpus, args.voc, args.semantic, val_split=0.1)

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

    # Convert targets to PyTorch format
    targets_flat = torch.tensor([item for sublist in batch["targets"] for item in sublist], dtype=torch.int32).to(device)
    target_lengths = torch.tensor([len(target) for target in batch["targets"]], dtype=torch.int32).to(device)

    # Forward pass
    logits, seq_lengths_out = model(inputs, seq_lengths)
    log_probs = logits.log_softmax(2)  # Shape: [T, N, C]

    # CTC loss expects input_lengths as a tensor
    input_lengths = seq_lengths_out.to(torch.int32)

    # PyTorch CTC loss
    print("Input lengths:", input_lengths)
    print("Target lengths:", target_lengths)

    loss = ctc_loss(
        log_probs,
        targets_flat,
        input_lengths,
        target_lengths
    )
    print(loss)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    if epoch % 1000 == 0:
        print(f"Loss value at epoch {epoch}: {loss.item()}")
        print("Validating...")
        model.eval()
        
        # Initialize validation metrics
        val_ed = 0
        val_len = 0
        val_count = 0
        
        with torch.no_grad():
            validation_size = primus.getValidationSize()
            
            for val_idx in range(0, validation_size, batch_size):
                # Get batch of validation data
                validation_batch = primus.getValidationBatch(val_idx, batch_size, params)
                
                # Convert to tensors and move to device
                mini_batch_inputs = torch.tensor(
                    validation_batch["inputs"],
                    dtype=torch.float32,
                ).to(device)
                
                mini_batch_seq_lengths = torch.tensor(
                    validation_batch["seq_lengths"],
                    dtype=torch.int32,
                ).to(device)
                
                # Forward pass
                logits, seq_lengths_out = model(
                    mini_batch_inputs, mini_batch_seq_lengths
                )
                
                # Greedy decoding
                log_probs = logits.log_softmax(2)  # Shape: [T, N, C]
                predictions = torch.argmax(log_probs, dim=2).transpose(0, 1)  # Shape: [N, T]

                # Remove duplicates and blank tokens (CTC decoding logic)
                blank_token = params["vocabulary_size"]
                decoded_predictions = []
                for seq in predictions:
                    decoded_seq = []
                    previous_token = None
                    for token in seq:
                        if token != previous_token and token != blank_token:
                            decoded_seq.append(token.item())
                        previous_token = token
                    decoded_predictions.append(decoded_seq)
                
                # Calculate metrics
                current_batch_size = len(decoded_predictions)
                for i, pred in enumerate(decoded_predictions):
                    ed = ctc_utils.edit_distance(
                        pred, validation_batch["targets"][i]
                    )
                    val_ed += ed
                    val_len += len(validation_batch["targets"][i])
                    val_count += 1
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
                
                # Print progress
                print(f"Processed {val_idx + current_batch_size}/{validation_size} validation samples", end='\r')
                
        print(
            f"\n[Epoch {epoch}] SER: {100.0 * val_ed / val_len:.2f}% "
            f"({val_ed / val_count:.2f} average edit distance)."
        )

        # Save model
        torch.save(model.state_dict(), f"{args.save_model}_epoch_{epoch}.pth")
        print("Model saved.")
