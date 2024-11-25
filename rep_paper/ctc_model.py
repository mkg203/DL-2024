import torch
import torch.nn as nn
import torch.nn.functional as F


def leaky_relu(features, alpha=0.2):
    return F.leaky_relu(features, negative_slope=alpha)


def default_model_params(img_height, vocabulary_size):
    params = dict()
    params["img_height"] = img_height
    params["img_width"] = None
    params["batch_size"] = 16
    params["img_channels"] = 1
    params["conv_blocks"] = 4
    params["conv_filter_n"] = [32, 64, 128, 256]
    params["conv_filter_size"] = [3, 3, 3, 3]
    params["conv_pooling_size"] = [2, 2, 2, 2]
    params["rnn_units"] = 512
    params["rnn_layers"] = 2
    params["vocabulary_size"] = vocabulary_size
    return params


class CTCCRNN(nn.Module):
    def __init__(self, params):
        super(CTCCRNN, self).__init__()
        self.params = params

        # Create sequential convolutional blocks
        conv_layers = []
        in_channels = params["img_channels"]

        for i in range(params["conv_blocks"]):
            # Convolutional layer
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=params["conv_filter_n"][i],
                    kernel_size=params["conv_filter_size"][i],
                    padding="same",
                )
            )
            # Batch normalization
            conv_layers.append(nn.BatchNorm2d(params["conv_filter_n"][i]))
            # Leaky ReLU
            conv_layers.append(nn.LeakyReLU(0.2))
            # Max pooling
            conv_layers.append(
                nn.MaxPool2d(
                    kernel_size=params["conv_pooling_size"][i],
                    stride=params["conv_pooling_size"][i],
                )
            )

            in_channels = params["conv_filter_n"][i]

        self.conv_layers = nn.Sequential(*conv_layers)

        # Bidirectional LSTM layers
        self.rnn_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=(
                        params["conv_filter_n"][-1]
                        if i == 0
                        else params["rnn_units"] * 2
                    ),
                    hidden_size=params["rnn_units"],
                    bidirectional=True,
                    batch_first=True,
                )
                for i in range(params["rnn_layers"])
            ]
        )

        # Final fully connected layer
        self.fc = nn.Linear(
            params["rnn_units"] * 2, params["vocabulary_size"] + 1  # +1 for CTC blank
        )

    def forward(self, x):
        # Apply convolutional layers
        conv_output = self.conv_layers(x)

        # Prepare for RNN (B, C, H, W) -> (B, W, H*C)
        batch_size, channels, height, width = conv_output.size()
        rnn_input = conv_output.permute(0, 3, 1, 2)
        rnn_input = rnn_input.contiguous().view(batch_size, width, channels * height)

        # Apply RNN layers
        rnn_output = rnn_input
        for rnn_layer in self.rnn_layers:
            rnn_output, _ = rnn_layer(rnn_output)

        # Apply final fully connected layer
        logits = self.fc(rnn_output)

        # Log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs

    def get_seq_length(self, input_length):
        """Calculate sequence length after CNN layers"""
        seq_length = input_length
        for i in range(self.params["conv_blocks"]):
            seq_length = seq_length // self.params["conv_pooling_size"][i]
        return seq_length


def ctc_loss(logits, targets, seq_len, target_len):
    """
    Calculate CTC loss
    Args:
        logits: (B, T, C) log probabilities
        targets: Target sequences
        seq_len: Length of input sequences
        target_len: Length of target sequences
    """
    loss = F.ctc_loss(
        logits.transpose(0, 1),  # (T, B, C) required by CTC loss
        targets,
        seq_len,
        target_len,
        blank=0,
        reduction="mean",
    )
    return loss
