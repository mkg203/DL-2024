import torch
import torch.nn as nn
import torch.nn.functional as F


def leaky_relu(features, alpha=0.2):
    return torch.maximum(alpha * features, features)


class CTC_CRNN(nn.Module):
    def __init__(self, params):
        super(CTC_CRNN, self).__init__()

        self.params = params
        self.conv_blocks = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Number of input channels (e.g., 1 for grayscale images)
        input_channels = params["img_channels"]

        # Convolutional layers
        for i in range(params["conv_blocks"]):
            conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=params["conv_filter_n"][i],
                kernel_size=params["conv_filter_size"][i],
                padding=1,  # 'same' padding equivalent
            )
            bn = nn.BatchNorm2d(params["conv_filter_n"][i])
            pool = nn.MaxPool2d(kernel_size=params["conv_pooling_size"][i])
            self.conv_blocks.append(conv)
            self.batch_norms.append(bn)
            self.pools.append(pool)
            input_channels = params["conv_filter_n"][i]  # Update for the next block

        # Calculate RNN input size
        conv_height = params["img_height"] // (2 ** len(params["conv_pooling_size"]))
        self.rnn_input_size = params["conv_filter_n"][-1] * conv_height

        # RNN layers
        self.rnn_units = params["rnn_units"]
        self.rnn_layers = params["rnn_layers"]
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_units,
            num_layers=self.rnn_layers,
            bidirectional=True,
            batch_first=False,
        )

        # Fully connected output layer
        self.fc = nn.Linear(
            self.rnn_units * 2, params["vocabulary_size"] + 1
        )  # +1 for the CTC blank token

    def forward(self, x, seq_lengths):
        """
        Forward pass for CTC_CRNN
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            seq_lengths (torch.Tensor): Sequence lengths for CTC loss
        Returns:
            logits (torch.Tensor): Logits for CTC loss, shape [time_steps, batch_size, num_classes]
            seq_lengths (torch.Tensor): Adjusted sequence lengths
        """
        # Ensure input is in NCHW format
        if x.size(1) != self.params["img_channels"]:
            raise ValueError(
                f"Expected {self.params['img_channels']} channels, got {x.size(1)}"
            )

        # Apply convolutional layers
        for conv, bn, pool in zip(self.conv_blocks, self.batch_norms, self.pools):
            x = pool(F.leaky_relu(bn(conv(x))))

        # Prepare features for RNN
        b, c, h, w = x.size()
        x = x.permute(3, 0, 2, 1).contiguous()  # [width, batch, height, channels]
        x = x.view(w, b, -1)  # [width, batch, features]

        # RNN layers
        x, _ = self.lstm(x)

        # Fully connected layer
        logits = self.fc(x)

        return logits, seq_lengths


def default_model_params(img_height, vocabulary_size):
    params = {
        "img_height": img_height,
        "img_width": None,
        "batch_size": 16,
        "img_channels": 1,
        "conv_blocks": 4,
        "conv_filter_n": [32, 64, 128, 256],
        "conv_filter_size": [[3, 3], [3, 3], [3, 3], [3, 3]],
        "conv_pooling_size": [[2, 2], [2, 2], [2, 2], [2, 2]],
        "rnn_units": 512,
        "rnn_layers": 2,
        "vocabulary_size": vocabulary_size,
    }
    return params