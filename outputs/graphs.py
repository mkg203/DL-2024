#training_log_3_layer_135
import matplotlib.pyplot as plt


epochs = list(range(1, 135))  
training_loss = [
    4.3945, 3.8029, 3.1791, 2.6941, 2.3037, 1.9765, 1.7119, 1.5221, 1.3715, 1.2493,
    1.1401, 1.0535, 0.9712, 0.9019, 0.8347, 0.7736, 0.7245, 0.6687, 0.6342, 0.5915,
    0.5458, 0.5202, 0.4885, 0.4582, 0.4523, 0.4246, 0.3970, 0.3783, 0.3587, 0.3520,
    0.3338, 0.3310, 0.3057, 0.3015, 0.2807, 0.2734, 0.2643, 0.2560, 0.2391, 0.2326,
    0.2269, 0.2121, 0.2115, 0.1996, 0.1971, 0.1943, 0.1900, 0.1870, 0.1774, 0.1671,
    0.1753, 0.1699, 0.1546, 0.1501, 0.1482, 0.1549, 0.1439, 0.1453, 0.1402, 0.1356,
    0.1307, 0.1299, 0.1311, 0.1283, 0.1290, 0.1202, 0.1152, 0.1107, 0.1102, 0.1089,
    0.1041, 0.1059, 0.1030, 0.1031, 0.1008, 0.0987, 0.0987, 0.1026, 0.0956, 0.1013,
    0.0981, 0.0851, 0.0868, 0.0842, 0.0871, 0.0830, 0.0849, 0.0869, 0.0814, 0.0768,
    0.0763, 0.0812, 0.0810, 0.0734, 0.0706, 0.0726, 0.0724, 0.0747, 0.0712, 0.0647,
    0.0676, 0.0671, 0.0689, 0.0686, 0.0641, 0.0615, 0.0593, 0.0580, 0.0631, 0.0683,
    0.0596, 0.0573, 0.0583, 0.0541, 0.0593, 0.0547, 0.0613, 0.0533, 0.0510, 0.0532,
    0.0512, 0.0531, 0.0531, 0.0510, 0.0541, 0.0530, 0.0518, 0.0570, 0.0501, 0.0480,
    0.0487, 0.0447, 0.0487, 0.0458
]  # List of training loss values (length 134)
validation_loss = [
    4.1117, 3.4746, 2.9161, 2.5256, 2.1769, 1.8846, 1.6375, 1.5002, 1.3647, 1.2835,
    1.1179, 1.0278, 0.9427, 0.8875, 0.8211, 0.7981, 0.7002, 0.6593, 0.6422, 0.5692,
    0.5407, 0.5119, 0.5060, 0.4749, 0.4565, 0.4301, 0.4171, 0.3843, 0.3972, 0.3782,
    0.3452, 0.3399, 0.3524, 0.3202, 0.3062, 0.2943, 0.2935, 0.2864, 0.2802, 0.2705,
    0.2665, 0.2552, 0.2581, 0.2391, 0.2414, 0.2524, 0.2526, 0.2392, 0.2268, 0.2296,
    0.2494, 0.2224, 0.2050, 0.2096, 0.2304, 0.2038, 0.2020, 0.2164, 0.1930, 0.2050,
    0.2032, 0.1996, 0.2072, 0.2060, 0.1953, 0.1879, 0.1875, 0.1836, 0.1919, 0.1813,
    0.1784, 0.1878, 0.1844, 0.1802, 0.1766, 0.1767, 0.1749, 0.1762, 0.1835, 0.1785,
    0.1801, 0.1684, 0.1701, 0.1802, 0.1771, 0.1735, 0.1730, 0.1723, 0.1634, 0.1803,
    0.1762, 0.1689, 0.1583, 0.1623, 0.1668, 0.1762, 0.1666, 0.1668, 0.1654, 0.1740,
    0.1702, 0.1725, 0.1646, 0.1603, 0.1673, 0.1657, 0.1643, 0.1695, 0.1612, 0.1637,
    0.1631, 0.1653, 0.1586, 0.1641, 0.1643, 0.1692, 0.1584, 0.1695, 0.1648, 0.1601,
    0.1588, 0.1656, 0.1651, 0.1638, 0.1619, 0.1693, 0.1650, 0.1696, 0.1639, 0.1655,
    0.1614, 0.1620, 0.1674
]  # List of validation loss values (length 133)


validation_epochs = epochs[:len(validation_loss)]

plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, label='Training Loss', marker='o')
plt.plot(validation_epochs, validation_loss, label='Validation Loss', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


import re
import matplotlib.pyplot as plt
import os


log_file_path = r"C:\Users\abhiv\Downloads\training_log_2_257_at.log"

os.path.exists(log_file_path)
with open(log_file_path, 'r') as file:
    log_data = file.read()
epochs = []
training_losses = []
validation_losses = []

# Regular expressions to match the required lines
epoch_pattern = re.compile(r'Epoch\s+(\d+)/\d+')
training_loss_pattern = re.compile(r'Training Loss:\s*([0-9.]+)')
validation_loss_pattern = re.compile(r'Validation Loss:\s*([0-9.]+)')

# Split the log data into lines for processing
lines = log_data.strip().split('\n')

current_epoch = None

for line in lines:
    # Check for epoch line
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        epochs.append(current_epoch)
        continue  # Move to the next line

    # Check for training loss
    train_match = training_loss_pattern.search(line)
    if train_match and current_epoch is not None:
        training_loss = float(train_match.group(1))
        training_losses.append(training_loss)
        continue  # Move to the next line

    # Check for validation loss
    val_match = validation_loss_pattern.search(line)
    if val_match and current_epoch is not None:
        validation_loss = float(val_match.group(1))
        validation_losses.append(validation_loss)
        continue  # Move to the next line

# Verify that all lists have the same length
if not (len(epochs) == len(training_losses) == len(validation_losses)):
    print("Warning: Mismatch in the lengths of the extracted data lists.")
    print(f"Epochs: {len(epochs)}, Training Losses: {len(training_losses)}, Validation Losses: {len(validation_losses)}")

# Plotting the Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, training_losses, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, validation_losses, label='Validation Loss', color='orange', marker='x')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



