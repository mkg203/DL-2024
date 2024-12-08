import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image', dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model (.meta file).')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

# Enable TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

# Reset graph (for compatibility mode)
tf.compat.v1.reset_default_graph()

# Start a session
sess = tf.compat.v1.Session()

# Load the dictionary
with open(args.voc_file, 'r') as dict_file:
    dict_list = dict_file.read().splitlines()
int2word = {idx: word for idx, word in enumerate(dict_list)}

# Restore the pretrained model
saver = tf.compat.v1.train.import_meta_graph(args.model)
saver.restore(sess, args.model[:-5])  # Remove the '.meta' extension to restore the model

# Get graph and tensors
graph = tf.compat.v1.get_default_graph()
input_tensor = graph.get_tensor_by_name("model_input:0")
seq_len_tensor = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob_tensor = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.compat.v1.get_collection("logits")[0]

# Extract constants
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

# Image preprocessing
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
image = ctc_utils.resize(image, HEIGHT)
image = ctc_utils.normalize(image)
image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions

seq_lengths = [image.shape[2] // WIDTH_REDUCTION]

# Perform prediction
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len_tensor)
prediction = sess.run(decoded, feed_dict={
    input_tensor: image,
    seq_len_tensor: seq_lengths,
    rnn_keep_prob_tensor: 1.0,
})

# Convert predictions to strings
str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
for w in str_predictions[0]:
    print(int2word[w], end='\t')
