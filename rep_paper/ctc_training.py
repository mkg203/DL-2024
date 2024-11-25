import torch
import torch.nn as nn
import torch.optim as optim
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse
import os

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

# Load primus
primus = CTC_PriMuS(args.corpus, args.set, args.voc, args.semantic, val_split=0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height, primus.vocabulary_size)
max_epochs = 64000
dropout = 0.5

# Model
model = ctc_model.ctc_crnn(params).cuda()
criterion = nn.CTCLoss(blank=0, reduction="mean").cuda()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(max_epochs):
    batch = primus.nextBatch(params)
    inputs = torch.tensor(batch["inputs"]).cuda()
    seq_len = torch.tensor(batch["seq_lengths"]).cuda()
    targets = ctc_utils.sparse_tuple_from(batch["targets"])

    model.train()
    optimizer.zero_grad()
    outputs, decoded = model(inputs)

    loss = criterion(outputs.log_softmax(2), targets[0], seq_len, targets[1])
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        # VALIDATION
        print("Loss value at epoch " + str(epoch) + ":" + str(loss.item()))
        print("Validating...")

        validation_batch, validation_size = primus.getValidation(params)

        val_idx = 0
        val_ed = 0
        val_len = 0
        val_count = 0

        while val_idx < validation_size:
            mini_batch_inputs = torch.tensor(
                validation_batch["inputs"][val_idx : val_idx + params["batch_size"]]
            ).cuda()
            mini_batch_seq_len = torch.tensor(
                validation_batch["seq_lengths"][
                    val_idx : val_idx + params["batch_size"]
                ]
            ).cuda()

            with torch.no_grad():
                prediction = model(mini_batch_inputs)

            str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)

            for i in range(len(str_predictions)):
                ed = ctc_utils.edit_distance(
                    str_predictions[i], validation_batch["targets"][val_idx + i]
                )
                val_ed += ed
                val_len += len(validation_batch["targets"][val_idx + i])
                val_count += 1

            val_idx += params["batch_size"]

        print(
            "[Epoch "
            + str(epoch)
            + "] "
            + str(1.0 * val_ed / val_count)
            + " ("
            + str(100.0 * val_ed / val_len)
            + " SER) from "
            + str(val_count)
            + " samples."
        )
        print("Saving the model...")
        torch.save(model.state_dict(), args.save_model + "_epoch_" + str(epoch) + ".pt")
        print("------------------------------")
