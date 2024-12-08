import tensorflow as tf

def leaky_relu(features, alpha=0.2, name=None):
    with tf.name_scope(name or "LeakyRelu"):
        features = tf.convert_to_tensor(features, name="features")
        alpha = tf.convert_to_tensor(alpha, name="alpha")
        return tf.maximum(alpha * features, features)



# params["height"] = height of the input image
# params["width"] = width of the input image

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 8
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [[3, 3], [3, 3], [3, 3], [3, 3]]
    params['conv_pooling_size'] = [[2, 2], [2, 2], [2, 2], [2, 2]]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params


def ctc_crnn(params):
    input = tf.keras.Input(
        shape=(params['img_height'], None, params['img_channels']),
        dtype=tf.float32,
        name='model_input'
    )
    
    x = input
    width_reduction = 1
    height_reduction = 1
    
    for i in range(params['conv_blocks']):
        x = tf.keras.layers.Conv2D(
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = leaky_relu(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=params['conv_pooling_size'][i],
            strides=params['conv_pooling_size'][i])(x)
        
        width_reduction *= params['conv_pooling_size'][i][1]
        height_reduction *= params['conv_pooling_size'][i][0]
    
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] // height_reduction)
    features = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch, width, height, channels]
    features = tf.reshape(features, [tf.shape(features)[0], -1, feature_dim])  # [batch, width, features]
    
    lstm_input = tf.keras.layers.Input(tensor=features)
    seq_len_input = tf.keras.Input(shape=(None,), dtype=tf.int32, name='seq_lengths')
    rnn_keep_prob_input = tf.keras.Input(shape=(), dtype=tf.float32, name="keep_prob")

    for _ in range(params['rnn_layers']):
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True, dropout=1 - rnn_keep_prob_input))(lstm_input)
    
    logits = tf.keras.layers.Dense(params['vocabulary_size'] + 1, activation=None)(lstm)

    seq_len = seq_len_input
    targets = tf.sparse.SparseTensor(
        indices=tf.keras.layers.Input(name="target_indices", shape=(None, 2), dtype=tf.int64),
        values=tf.keras.layers.Input(name="target_values", shape=(None,), dtype=tf.int32),
        dense_shape=tf.keras.layers.Input(name="target_dense_shape", shape=(2,), dtype=tf.int64),
    )
    
    ctc_loss = tf.keras.backend.ctc_batch_cost(y_true=targets, y_pred=logits, input_length=seq_len, label_length=tf.constant(0))
    loss = tf.reduce_mean(ctc_loss)
    
    decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
    
    return input, seq_len, targets, decoded, loss, rnn_keep_prob_input
