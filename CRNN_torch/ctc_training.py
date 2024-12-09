import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from primus import CTC_PriMuS
import ctc_utils
from ctc_model import CTC_CRNN, default_model_params  # Assuming the PyTorch model and params
import argparse

# Create a unique log file
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


parser = argparse.ArgumentParser(description="Train model.")
parser.add_argument("-corpus", dest="corpus", type=str, required=True, help="Path to the corpus.")
parser.add_argument("-set", dest="set", type=str, required=True, help="Path to the set file.")
# parser.add_argument("-save_model", dest="save_model", type=str, required=True, help="Path to save the model.")
parser.add_argument("-vocabulary", dest="voc", type=str, required=True, help="Path to the vocabulary file.")
parser.add_argument("-semantic", dest="semantic", action="store_true", default=False)
args = parser.parse_args()

# Initialize Dataset
primus = CTC_PriMuS(args.set, args.corpus, args.voc, args.semantic, val_split=0.1)

# Parameterization
img_height = 128
params = default_model_params(img_height, primus.vocabulary_size)
max_epochs = 64000
dropout = 0.5
save_interval = 5000  # Change the model saving interval here

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTC_CRNN(params).to(device)
criterion = nn.CTCLoss(blank=params["vocabulary_size"]).to(device)
optimizer = optim.Adam(model.parameters())

# Logging Start
logging.info("Training started")
logging.info(f"Parameters: {params}")
logging.info(f"Device: {device}")

def prepare_input(batch):
    """
    Prepares inputs, sequence lengths, and targets for PyTorch training.
    """
    # Convert inputs to tensor
    inputs = torch.tensor(batch["inputs"], dtype=torch.float32).to(device)
    
    # Sequence lengths
    seq_lengths = torch.tensor(batch["seq_lengths"], dtype=torch.int32).to(device)
    
    # Flatten targets and calculate target lengths
    targets = batch["targets"]
    targets_flat = torch.tensor([item for sublist in targets for item in sublist], dtype=torch.int32).to(device)
    target_lengths = torch.tensor([len(seq) for seq in targets], dtype=torch.int32).to(device)
    
    return inputs, seq_lengths, targets_flat, target_lengths


# Training loop
for epoch in range(max_epochs):
    model.train()

    # Fetch Batch
    batch = primus.nextBatch(params)

    # Prepare inputs and targets
    inputs, seq_lengths, targets_flat, target_lengths = prepare_input(batch)

    # Forward pass
    logits, output_lengths = model(inputs, seq_lengths)
    logits = logits.log_softmax(2)  # Apply log softmax for CTC Loss

    # Calculate CTC loss
    loss = criterion(
        logits,          # [T, N, C]
        targets_flat,    # Targets (flattened)
        output_lengths,  # Lengths of logits
        target_lengths   # Lengths of targets
    )

    # Backward pass and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs (similar to TensorFlow validation prints)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
    if epoch % 1000 == 0:
        logging.info(f"Loss value at epoch {epoch}: {loss.item()}")
        logging.info("Validating...")

        model.eval()
        with torch.no_grad():
            validation_batch, validation_size = primus.getValidation(params)
            val_idx = 0
            val_ed, val_len, val_count = 0, 0, 0

            while val_idx < validation_size:
                batch_start = val_idx
                batch_end = val_idx + params["batch_size"]

                inputs = torch.tensor(
                    validation_batch["inputs"][batch_start:batch_end],
                    dtype=torch.float32,
                ).to(device)
                seq_lengths = torch.tensor(
                    validation_batch["seq_lengths"][batch_start:batch_end],
                    dtype=torch.int32,
                ).to(device)

                logits, _ = model(inputs, seq_lengths)
                decoded = model.greedy_decoder(logits)

                str_predictions = ctc_utils.sparse_tensor_to_strs(decoded)
                for i, pred in enumerate(str_predictions):
                    ed = ctc_utils.edit_distance(
                        pred, validation_batch["targets"][val_idx + i]
                    )
                    val_ed += ed
                    val_len += len(validation_batch["targets"][val_idx + i])
                    val_count += 1

                val_idx += params["batch_size"]

            ser = 100.0 * val_ed / val_len
            logging.info(
                f"[Epoch {epoch}] SER: {ser:.2f}% from {val_count} samples."
            )


        # Validation and Model Saving
        if epoch % save_interval == 0:
            # Save the Model
            model_path = f"model/epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved at {model_path}")
            logging.info("------------------------------")
