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
parser.add_argument("-corpus", dest="corpus", type=str, required=True, help="Path to the corpus.")
parser.add_argument("-set", dest="set", type=str, required=True, help="Path to the set file.")
# parser.add_argument("-save_model", dest="save_model", type=str, required=True, help="Path to save the model.")
parser.add_argument("-vocabulary", dest="voc", type=str, required=True, help="Path to the vocabulary file.")
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

ctc_loss = nn.CTCLoss(blank=params["vocabulary_size"], reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Added learning rate

def prepare_input(batch):
    # Convert inputs to NCHW format
    inputs = torch.tensor(batch["inputs"], dtype=torch.float32).to(device)
    # inputs = inputs.permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Prepare sequence lengths
    seq_lengths = torch.tensor(batch["seq_lengths"], dtype=torch.int32).to(device)
    
    # Prepare targets in sparse format
    targets = batch["targets"]
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.int32).to(device)
    targets_flat = torch.tensor([item for sublist in targets for item in sublist], dtype=torch.int32).to(device)
    
    return inputs, seq_lengths, targets_flat, target_lengths

# Training loop
max_epochs = 64000
batch_size = params["batch_size"]

for epoch in range(max_epochs):
    model.train()
    batch = primus.nextBatch(params)
    
    # Prepare inputs and targets
    inputs, seq_lengths, targets_flat, target_lengths = prepare_input(batch)
    
    # Forward pass
    optimizer.zero_grad()
    
    # Get logits and output sequence lengths
    logits, output_lengths = model(inputs, seq_lengths)
    
    # Ensure output_lengths are valid for CTC loss
    output_lengths = output_lengths.clamp(min=target_lengths.max().item())
    
    # Calculate CTC loss
    try:
        loss = ctc_loss(
            logits.log_softmax(2),  # [T, N, C]
            targets_flat,
            output_lengths,
            target_lengths
        )
        
        # Check for invalid loss values
        if torch.isfinite(loss):
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        else:
            print(f"Warning: Invalid loss value at epoch {epoch}: {loss.item()}")
            continue
            
    except RuntimeError as e:
        print(f"Error during training at epoch {epoch}: {str(e)}")
        continue

    # Validation
    if epoch % 1000 == 0:
        model.eval()
        print(f"\nValidating at epoch {epoch}...")
        
        val_ed = 0
        val_len = 0
        val_count = 0
        
        with torch.no_grad():
            validation_size = primus.getValidationSize()
            
            for val_idx in range(0, validation_size, batch_size):
                validation_batch = primus.getValidationBatch(val_idx, batch_size, params)
                
                # Prepare validation batch
                val_inputs, val_seq_lengths, _, _ = prepare_input(validation_batch)
                
                # Forward pass
                logits, output_lengths = model(val_inputs, val_seq_lengths)
                
                # Greedy decoding
                predictions = torch.argmax(logits.log_softmax(2), dim=2).transpose(0, 1)
                
                # Decode predictions
                decoded_predictions = []
                for pred_seq, length in zip(predictions, output_lengths):
                    pred_seq = pred_seq[:length]  # Trim to actual length
                    decoded_seq = []
                    previous = None
                    for token in pred_seq:
                        token = token.item()
                        if token != previous and token != params["vocabulary_size"]:
                            decoded_seq.append(token)
                        previous = token
                    decoded_predictions.append(decoded_seq)

                # print(decoded_predictions)
                
                # Calculate metrics
                current_batch_size = len(decoded_predictions)
                for i, pred in enumerate(decoded_predictions):
                    if i < len(validation_batch["targets"]):
                        ed = ctc_utils.edit_distance(pred, validation_batch["targets"][i])
                        val_ed += ed
                        val_len += len(validation_batch["targets"][i])
                        val_count += 1
                
                print(f"Processed {val_idx + current_batch_size}/{validation_size} validation samples", end='\r')
        
        if val_count > 0:
            print(f"\n[Epoch {epoch}] SER: {100.0 * val_ed / val_len:.2f}% "
                  f"({val_ed / val_count:.2f} average edit distance).")
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item() if torch.isfinite(loss) else None,
            }, f"model_saves/CRNN_epoch_{epoch}.pth")
            print("Model saved.")
