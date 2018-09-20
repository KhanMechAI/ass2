import tensorflow as tf
import string
import numpy as np

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 200  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
INPUT_SIZE = [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE]

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    
    processed_review = []
    string_translator = str.maketrans('','', string.punctuation)
    review = review.translate(string_translator).split(' ')

    for word in review:
        if word:
            if word in stop_words:
                next
            else:
                processed_review.append(word.lower())
    return processed_review
    
#Testing preprocess
# test_string1 = '...sdv.sdvj fv.advnaf  v  ...  favlkndfvav...v adfv ! ? ! $ sdfbvkjbnddfv '
# test_string2 = 'the boy and the cat were alone in the woods having a blast of a time. Alas did a tree fall and crtush the poor ca=t, and the boy was enver the same again'
# preprocess(test_string1)
# preprocess(test_string2)

def simple_sigmoid(bias, weights, inputs):
    term_1 = tf.matmul(weights[0], inputs[0])
    term_2 = tf.matmul(weights[1], inputs[1])
    return tf.sigmoid(tf.add(bias, term_1, term_2))

def simple_tanh(bias, weights, inputs):
    term_1 = tf.matmul(weights[0], inputs[0])
    term_2 = tf.matmul(weights[1], inputs[1])
    return tf.tanh(tf.add(bias, term_1, term_2))

# def lstm(x_t, h_t_1, c_t_1):
#     #TODO: figure out input size variable
#     #Forget gate
#     Wf = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     Uf = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     bf = tf.Variable(tf.random_uniform(shape=[1, INPUT_SIZE], maxval=1, minval=0), validate_shape=False)

#     f_t = simple_sigmoid(bf,[Wf, Uf], [x_t, h_t_1])

#     #Remember gate
#     Wi = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     Ui = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     bi = tf.Variable(tf.random_uniform(shape=[1, INPUT_SIZE], maxval=1, minval=0), validate_shape=False)

#     i_t = simple_sigmoid(bi,[Wi, Ui], [x_t, h_t_1])

#     Wg = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     Ug = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     bg = tf.Variable(tf.random_uniform(shape=[1, INPUT_SIZE], maxval=1, minval=0), validate_shape=False)

#     g_t = simple_sigmoid(bg,[Wg, Ug], [x_t, h_t_1])

#     #Selection
#     Wo = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     Uo = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE, INPUT_SIZE], maxval=1), validate_shape=False)

#     bo = tf.Variable(tf.random_uniform(shape=[1, INPUT_SIZE], maxval=1, minval=0), validate_shape=False)

#     o_t = simple_sigmoid(bo,[Wo, Uo], [x_t, h_t_1])
#     #State

#     c_t = tf.add(tf.tensordot(c_t_1, f_t), tf.tensordot(i_t, g_t))

#     #Outputs
#     h_t = tf.tensordot(tf.tanh(c_t), o_t)

#     return c_t, h_t


# def rnn_lstm(x, w, b):
#     hidden_size_1 = 256
#     hidden_size_2 = 256
#     multi_rnn = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_size_1),rnn.BasicLSTMCell(hidden_size_2)])

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    num_classes = 2
    num_units = 320
    num_layers = 4
    dkp = tf.Variable(0.5)
    dropout_keep_prob = tf.placeholder_with_default(dkp,shape=[],name="dropout_keep_prob")
    LEARNING_RATE = 0.001

    input_data = tf.placeholder(dtype=tf.float32, shape=INPUT_SIZE, name="input_data")

    labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, num_classes], name="labels")


    lstm_rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
    num_units=num_units)


    lstm_c_fw = [tf.contrib.rnn.DropoutWrapper(lstm_rnn_cell, output_keep_prob=dropout_keep_prob)]


    lstm_c_bw = [tf.contrib.rnn.DropoutWrapper(lstm_rnn_cell, output_keep_prob=dropout_keep_prob)]


    #tf.transpose(input_data, (1,0,2))
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_c_fw,cells_bw=lstm_c_bw,inputs=input_data,dtype=tf.float32, time_major=False)


    dense_out = tf.layers.Dense(
    num_classes, activation=None, 
    kernel_initializer=tf.orthogonal_initializer())

    logits = dense_out(outputs[:,-1,:])


    predits=tf.nn.softmax(logits)
    
    # loss=tf.nn.softmax_cross_entropy_with_logits_v2(
        # logits=predits, labels=labels, name='loss',dim=0)
    loss = tf.reduce_sum(-tf.reduce_sum(tf.multiply(labels ,tf.log(predits)),1, name="loss"))

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='optimizer').minimize(loss)

    temp = tf.equal(tf.argmax(predits, 1), tf.argmax(labels, 1))

    Accuracy = tf.reduce_mean(tf.cast(temp, tf.float32), name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
