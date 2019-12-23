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




