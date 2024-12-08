import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import logging


df = pd.read_csv("../data.csv")
df_filtered = df[["agnostic", "semantic"]]

agn_vocab_file = "../agnostic_vocab.txt"
sem_vocab_file = "../semantic_vocab.txt"

with open(agn_vocab_file, "r") as file:
    agn_vocab = file.read().splitlines()
with open(sem_vocab_file, "r") as file:
    sem_vocab = file.read().splitlines()


df_pre = df_filtered.applymap(lambda x: x.split("\t")[0:-1])

train_data, temp_data = train_test_split(df_pre, test_size=0.2, random_state=42)

validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(validation_data)}")
print(f"Test size: {len(test_data)}")


def transform_row(row):
    return {
        "agnostic": " ".join(row["agnostic"]),  # Convert the list to a string
        "semantic": " ".join(row["semantic"]),  # Convert the list to a string
        "agnostic_tokens": ["<sos>"]
        + row["agnostic"]
        + ["<eos>"],  # Add <sos> and <eos>
        "semantic_tokens": ["<sos>"]
        + row["semantic"]
        + ["<eos>"],  # Add <sos> and <eos>
    }


train_data = train_data.apply(transform_row, axis=1).tolist()
validation_data = validation_data.apply(transform_row, axis=1).tolist()
test_data = test_data.apply(transform_row, axis=1).tolist()


# Setting the random seeds for reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# Vocabulary creation
class Vocabulary:
    def __init__(self, tokens_list):
        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
        self.token_to_index = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.index_to_token = {idx: tok for tok, idx in self.token_to_index.items()}
        self.build_vocab(tokens_list)

    def build_vocab(self, tokens_list):
        for tokens in tokens_list:
            for token in tokens:
                if token not in self.token_to_index:
                    idx = len(self.token_to_index)
                    self.token_to_index[token] = idx
                    self.index_to_token[idx] = token

    def __len__(self):
        return len(self.token_to_index)

    def token_to_id(self, token):
        return self.token_to_index.get(token, self.token_to_index["<unk>"])

    def id_to_token(self, idx):
        return self.index_to_token.get(idx, "<unk>")

    def tokens_to_ids(self, tokens):
        return [self.token_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token(idx) for idx in ids]


agnostic_vocab = Vocabulary(
    [d["agnostic_tokens"] for d in train_data + validation_data + test_data]
)
semantic_vocab = Vocabulary(
    [d["semantic_tokens"] for d in train_data + validation_data + test_data]
)


# Dataset definition
class MusicDataset(Dataset):
    def __init__(self, data, agnostic_vocab, semantic_vocab):
        self.data = data
        self.agnostic_vocab = agnostic_vocab
        self.semantic_vocab = semantic_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        agnostic_tokens = self.data[idx]["agnostic_tokens"]
        semantic_tokens = self.data[idx]["semantic_tokens"]
        agnostic_ids = self.agnostic_vocab.tokens_to_ids(agnostic_tokens)
        semantic_ids = self.semantic_vocab.tokens_to_ids(semantic_tokens)
        return torch.tensor(agnostic_ids), torch.tensor(semantic_ids)


# Create datasets and dataloaders
train_dataset = MusicDataset(train_data, agnostic_vocab, semantic_vocab)
validation_dataset = MusicDataset(validation_data, agnostic_vocab, semantic_vocab)
test_dataset = MusicDataset(test_data, agnostic_vocab, semantic_vocab)

batch_size = 35

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
)
validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
)


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch_size, hidden_dim]
        # hidden: [n_layers, batch_size, hidden_dim]
        # cell: [n_layers, batch_size, hidden_dim]
        return outputs, (hidden, cell)


# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [n_layers, batch_size, hidden_dim]
        # encoder_outputs: [src_len, batch_size, hidden_dim]

        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Repeat hidden state for each source token
        hidden = (
            hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        )  # [batch_size, src_len, hidden_dim]

        # Calculate energy
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs.permute(1, 0, 2)), dim=2))
        )  # [batch_size, src_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        # Softmax over attention weights
        return nn.functional.softmax(attention, dim=1)


# Decoder with attention
class Decoder(nn.Module):
    def __init__(
        self, output_dim, embedding_dim, hidden_dim, n_layers, dropout, attention
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hidden_dim]
        # cell: [n_layers, batch_size, hidden_dim]
        # encoder_outputs: [src_len, batch_size, hidden_dim]

        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, embedding_dim]

        # Attention
        attention_weights = self.attention(
            hidden, encoder_outputs
        )  # [batch_size, src_len]
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]

        # Weighted sum of encoder outputs
        encoder_outputs = encoder_outputs.permute(
            1, 0, 2
        )  # [batch_size, src_len, hidden_dim]
        weighted = torch.bmm(
            attention_weights, encoder_outputs
        )  # [batch_size, 1, hidden_dim]
        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, hidden_dim]

        # Combine embedded input and weighted encoder context
        rnn_input = torch.cat(
            (embedded, weighted), dim=2
        )  # [1, batch_size, embedding_dim + hidden_dim]

        # Pass through RNN
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Final output prediction
        prediction = self.fc_out(
            torch.cat((output.squeeze(0), weighted.squeeze(0)), dim=1)
        )  # [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Update: Get encoder_outputs
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


# Model initialization
input_dim = len(agnostic_vocab)
output_dim = len(semantic_vocab)
embedding_dim = 128
hidden_dim = 256
n_layers = 2
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = Attention(hidden_dim)
encoder = Encoder(input_dim, embedding_dim, hidden_dim, n_layers, dropout)
decoder = Decoder(output_dim, embedding_dim, hidden_dim, n_layers, dropout, attention)
model = Seq2Seq(encoder, decoder, device).to(device)


# Training setup
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=semantic_vocab.token_to_id("<pad>"))

# Setup logging
logging.basicConfig(
    filename="training_log.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src, trg = zip(*batch)
        src = nn.utils.rnn.pad_sequence(
            src, padding_value=agnostic_vocab.token_to_id("<pad>")
        ).to(device)
        trg = nn.utils.rnn.pad_sequence(
            trg, padding_value=semantic_vocab.token_to_id("<pad>")
        ).to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Batch {i+1}/{len(data_loader)}: Loss {loss.item():.4f}", end="\r")
    print()  # Move to the next line after the epoch
    return epoch_loss / len(data_loader)


# Validation function
def validate_fn(model, data_loader, criterion, agnostic_vocab, semantic_vocab):
    model.eval()
    epoch_loss = 0
    total_sequences = 0
    total_symbols = 0
    incorrect_sequences = 0
    incorrect_symbols = 0

    with torch.no_grad():
        for batch in data_loader:
            src, trg = zip(*batch)
            src = nn.utils.rnn.pad_sequence(
                src, padding_value=agnostic_vocab.token_to_id("<pad>")
            ).to(device)
            trg = nn.utils.rnn.pad_sequence(
                trg, padding_value=semantic_vocab.token_to_id("<pad>")
            ).to(device)

            # Forward pass
            output = model(
                src, trg, teacher_forcing_ratio=0
            )  # No teacher forcing during validation
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            # Convert predictions to token IDs
            predicted_ids = output.argmax(dim=1).view(-1)
            target_ids = trg.view(-1)

            # Calculate sequence-level errors
            for pred_seq, true_seq in zip(
                predicted_ids.split(trg.shape[0] // len(batch)),
                target_ids.split(trg.shape[0] // len(batch)),
            ):
                total_sequences += 1
                total_symbols += len(true_seq)
                if not torch.equal(pred_seq, true_seq):
                    incorrect_sequences += 1
                    incorrect_symbols += (pred_seq != true_seq).sum().item()

    avg_loss = epoch_loss / len(data_loader)
    sequence_error_rate = incorrect_sequences / total_sequences
    symbol_error_rate = incorrect_symbols / total_symbols

    return avg_loss, sequence_error_rate, symbol_error_rate


# Testing function
def test_fn(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            src, trg = zip(*batch)
            src = nn.utils.rnn.pad_sequence(
                src, padding_value=agnostic_vocab.token_to_id("<pad>")
            ).to(device)
            trg = nn.utils.rnn.pad_sequence(
                trg, padding_value=semantic_vocab.token_to_id("<pad>")
            ).to(device)
            output = model(
                src, trg, teacher_forcing_ratio=0
            )  # No teacher forcing during testing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            predictions.append(output.argmax(-1))  # Store predicted tokens for analysis
    return epoch_loss / len(data_loader), predictions


# Execution loop
def execute(
    model,
    train_loader,
    validation_loader,
    test_loader,
    optimizer,
    criterion,
    n_epochs,
    clip,
    teacher_forcing_ratio=0.5,
):
    best_valid_loss = float("inf")
    for epoch in range(n_epochs):
        teacher_forcing_ratio = max(teacher_forcing_ratio * (1 - epoch / n_epochs), 0.1)
        print(f"Epoch {epoch + 1}/{n_epochs}")
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, clip, teacher_forcing_ratio
        )
        valid_loss, seq_er, sym_er = validate_fn(
            model, validation_loader, criterion, agnostic_vocab, semantic_vocab
        )
        print(f"Training Loss: {train_loss:.4f} | Validation Loss: {valid_loss:.4f}")
        print(f"Sequence error: {seq_er:.4f} | Symbol error {sym_er:.4f}")
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")
        logging.info(f"Training Loss: {train_loss:.4f}")
        logging.info(f"Validation Loss: {valid_loss:.4f}")
        logging.info(f"Sequence Error Rate: {seq_er:.4f}")
        logging.info(f"Symbol Error Rate: {sym_er:.4f}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_model.pt")  # Save the best model
            print("Model saved!")
    # Load the best model and test
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, predictions = test_fn(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    return predictions


def translate_random_test_example(
    model, test_data, agnostic_vocab, semantic_vocab, max_len=50
):
    # Randomly select a test example
    test_example = random.choice(test_data)
    input_tokens = test_example["agnostic_tokens"]
    expected_output_tokens = test_example["semantic_tokens"]

    # Translate using the model
    model.eval()
    with torch.no_grad():
        # Add <sos> and <eos> to the input string
        input_tokens_with_sos_eos = ["<sos>"] + input_tokens + ["<eos>"]
        input_ids = agnostic_vocab.tokens_to_ids(input_tokens_with_sos_eos)
        input_tensor = (
            torch.tensor(input_ids).unsqueeze(1).to(device)
        )  # Add batch dimension

        # Pass through the encoder
        hidden, cell = model.encoder(input_tensor)

        # Initialize the decoder with <sos> token
        trg_indexes = [semantic_vocab.token_to_id("<sos>")]
        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == semantic_vocab.token_to_id("<eos>"):
                break

        # Convert token IDs to tokens
        output_tokens = semantic_vocab.ids_to_tokens(
            trg_indexes[1:-1]
        )  # Exclude <sos> and <eos>

    # Print input, expected output, and model's output
    print(f"Input String: {input_tokens}\n\n")
    print(f"Expected Output: {expected_output_tokens}\n\n")
    print(f"Model Output: {output_tokens}")


if __name__ == "__main__":
    # Hyperparameters
    n_epochs = 1000
    clip = 1.0
    teacher_forcing_ratio = 0.8

    # Execute the training, validation, and testing process
    predictions = execute(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=n_epochs,
        clip=clip,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )
