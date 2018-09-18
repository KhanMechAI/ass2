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
    

def simple_sigmoid(bias, weights, inputs):
    term_1 = tf.matmul(weights[0], inputs[0])
    term_2 = tf.matmul(weights[1], inputs[1])
    return tf.sigmoid(tf.add(bias, term_1, term_2))

def simple_tanh(bias, weights, inputs):
    term_1 = tf.matmul(weights[0], inputs[0])
    term_2 = tf.matmul(weights[1], inputs[1])
    return tf.tanh(tf.add(bias, term_1, term_2))


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

    
    lstm_rnn = tf.contrib.cudnn_rnn.CudnnLSTM(
    num_layers=num_layers,
    num_units=num_units,
    input_mode='linear_input',
    direction='unidirectional',
    dropout=0.5,
    seed=np.random.randint(low=10000),
    dtype=tf.float32,
    kernel_initializer=None,
    bias_initializer=None,
    name=None)

    outputs,_ = lstm_rnn(inputs=input_data, training=True)

    dense_out = tf.layers.Dense(
    num_classes, activation=None, 
    kernel_initializer=tf.orthogonal_initializer())

    logits = dense_out(outputs[:,-1,:])

    # predits=tf.nn.softmax(logits)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels),name='loss')
            
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='optimiser').minimize(loss)

    temp = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    Accuracy = tf.reduce_mean(tf.cast(temp, tf.float32),name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
