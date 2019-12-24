######################### General Conversational Chatbot ##############################

################ Deep Learning ; Natural Language Processing ##########################

# Importing necessary libraries
import numpy as np
import tensorflow as tf
import re
import time


################ Data Preprocessing ##############

# Import dataset
lines = open("movie_lines.txt", encoding = "utf-8", errors = "ignore").read().split("\n")
conversations = open("movie_conversations.txt", encoding = "utf-8", errors = "ignore").read().split("\n")

# Mapping movie lines and ids
id_line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id_line[_line[0]] = _line[4]

# Creating a list of all the conversations
conversation_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(","))

# Mapping inputs and outputs
questions = []
answers = []

for conversation in conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id_line[conversation[i]])
        answers.append(id_line[conversation[i+1]])
        

# Cleaning the texts


def clean_text(text):    ## I will define a custom text cleaning function
    text = text.lower()  ## I will make all the texts lower case
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"why's", "why is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


### Cleaning the questions list
questions_cleaned = [clean_text(question) for question in questions]

### Cleaning the answers list
answers_cleaned = [clean_text(answer) for answer in answers]



## Creating a dictionary that maps each word to its number of occurrences
word2count = {}

for question in questions_cleaned:
    for word in question.split():
        if word in word2count:
            word2count[word] += 1
        else:
            word2count[word] = 1
            
for answer in answers_cleaned:
    for word in answer.split():
        if word in word2count:
            word2count[word] += 1
        else:
            word2count[word] = 1


## Creating two dictionaries that map the question words and answer words to unique integers
threshold_of_occurrence = 20
questionwords2int = {}
unique_integer = 0

for word, count in word2count.items():
    if count >= threshold_of_occurrence:
        questionwords2int[word] = unique_integer
        unique_integer += 1


answerwords2int = {}
unique_integer = 0
for word, count in word2count.items():
    if count >= threshold_of_occurrence:
        answerwords2int[word] = unique_integer
        unique_integer += 1
        
## Adding the last tokens (SOS, EOS) to the two dictionaries
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
    answerwords2int[token] = len(answerwords2int) + 1

## Creating the inverse dictionary of the answerwords2int dictionary
answerint2word = {_int: word for word, _int in answerwords2int.items()}


## Adding the EOS token to the end of every answer
for i in range(len(answers_cleaned)):
    answers_cleaned[i] += " <EOS>"


## Translating all the questions and the answers into integers
## and replacing all the words that were filtered out by <OUT>
questions_to_int = []
for question in questions_cleaned:
    integers = []
    for word in question.split():
        if word not in questionwords2int:
            integers.append(questionwords2int["<OUT>"])
        else:
            integers.append(questionwords2int[word])
    questions_to_int.append(integers)
    

answers_to_int = []
for answer in answers_cleaned:
    integers = []
    for word in answer.split():
        if word not in answerwords2int:
            integers.append(answerwords2int["<OUT>"])
        else:
            integers.append(answerwords2int[word])
    answers_to_int.append(integers)

## Sorting questions and answers by the length of questions (helps speed up the training)
questions_cleaned_sorted = []
answers_cleaned_sorted = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            questions_cleaned_sorted.append(questions_to_int[i[0]])
            answers_cleaned_sorted.append(answers_to_int[i[0]])




############## Building the Seq2Seq Model #####################################

## Creating placeholders for the inputs and the targets
def model_inputs():
    # The input and target are two-dimensional matrices with unique integers
    inputs = tf.placeholder(tf.int32, [None, None], name = "input")  
    targets = tf.placeholder(tf.int32, [None, None], name = "target")
    learning_rate = tf.placeholder(tf.float32, name = "learning_rate")  # store the learning rate 
    keep_proba = tf.placeholder(tf.float32, name = "keep_proba")  # controls the drop-out rate
    
    return inputs, targets, learning_rate, keep_proba

    

## Preprocessing the targets
########## targets must be in batches 
def preprocess_targets(targets, batch_size, word2int):
    SOS_matrix = tf.fill([batch_size, 1], word2int["<SOS>"])
    batched_answers = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])         # strided_slice function extracts a subset of the tensor 
    preprocessed_targets = tf.concat([SOS_matrix, batched_answers], axis = 1)
    
    return preprocessed_targets
    




###### Creating the Encoder RNN Model ########
def rnn_encoder(rnn_inputs, rnn_size, num_layers, keep_proba, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_proba)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state




### Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, 
                        output_function, keep_proba, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states = attention_states,
                                                                                                                                    attention_option = "bahdanau",
                                                                                                                                    num_units = decoder_cell.output_size)
                                                                                                                                    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_proba)
    return output_function(decoder_output_dropout)



### Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, 
                    num_words, decoding_scope, output_function, keep_proba, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states = attention_states,
                                                                        attention_option = "bahdanau",
                                                                        num_units = decoder_cell.output_size)
                                                                                                                                    
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, 
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions



### Creating the RNN Decoder
def rnn_decoder(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, 
                rnn_size, num_layers, word2int, keep_proba, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_proba)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_proba,
                                                   batch_size)
        
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int["<SOS>"],
                                           word2int["<EOS>"],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_proba,
                                           batch_size)
    return training_predictions, test_predictions


#### Putting together the Seq2Seq Model ########
def seq2seq_model(inputs, targets, keep_proba, batch_size, sequence_length, answers_num_words, 
                  questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, 
                  num_layers, questionwords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))

    encoder_state = rnn_encoder(encoder_embedded_input,
                                rnn_size,
                                num_layers,
                                keep_proba,
                                sequence_length)
    
    preprocessed_targets = preprocess_targets(targets,
                                              batch_size,
                                              questionwords2int)



    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, 
                                                               decoder_embedding_size], 0, 1))
    
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,
                                                    preprocessed_targets)
    training_predictions, test_predictions = rnn_decoder(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionwords2int,
                                                         keep_proba,
                                                         batch_size)
    
    return training_predictions, test_predictions



################### Training Seq2Seq Model ###############################################
##########################################################################################
    
## Setting the hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9 # the percent of which the learning_rate is decayed over the duration of the iterations
min_learning_rate = 0.0001
keep_probability = 0.5 # selected based on the best practice value suggested from Geoffrey Hinton's paper for hidden units



## Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

## Loading the model inputs
inputs, targets, learning_rate, keep_proba = model_inputs()

## Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length")

## Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

## Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_proba,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerwords2int),
                                                       len(questionwords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionwords2int)
                                                                   
                                                                   


## Setting up the Loss Error, the Optimizer and the Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))  # initialize weights to 1
                                                  
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(gradient_tensor, -5., 5.), gradient_variable) for gradient_tensor, gradient_variable in gradients if gradient_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)



## Padding the sequences with the <PAD> token
## Question: ['who' 'are' 'you', <PAD>, <PAD>, <PAD>, <PAD>]
## Answer: [<SOS> 'I' 'am' 'a' 'bot' '.' <EOS>, <PAD>, <PAD>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    padded_sequences = [sequence + [word2int["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
    return padded_sequences


## Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)//batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        
        # apply padding
        questions_in_batch_padded = np.array(apply_padding(questions_in_batch, questionwords2int))
        answers_in_batch_padded = np.array(apply_padding(answers_in_batch, answerwords2int))
        yield questions_in_batch_padded, answers_in_batch_padded


## Splitting the questions and answers into training and validation sets
training_validation_split_index = int(len(questions_cleaned_sorted) * 0.15)
training_questions = questions_cleaned_sorted[training_validation_split_index:]
training_answers = answers_cleaned_sorted[training_validation_split_index:]
validation_questions = questions_cleaned_sorted[:training_validation_split_index]
validation_answers = answers_cleaned_sorted[:training_validation_split_index]









