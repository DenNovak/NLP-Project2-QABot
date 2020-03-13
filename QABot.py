import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)

with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

print("Length of train data:", len(train_data))
print("Length of test data:", len(test_data))

vocab = set()
all_data = test_data + train_data

for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')
vocab_len = len(vocab) +1

max_story_len = max([len(data[0]) for data in all_data])
max_question_len =  max([len(data[1]) for data in all_data])

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len = max_story_len, max_question_len = max_question_len):
    '''
    INPUT:

    data: consisting of Stories,Queries,and Answers
    word_index: word index dictionary from tokenizer
    max_story_len: the length of the longest story
    max_question_len: length of the longest question


    OUTPUT:

    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.

    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    '''

    # STORIES = X
    X = []
    # QUESTIONS Xq
    Xq = []
    # Y CORRECT ANSWER
    Y = []

    for story, query, answer in data:

        # for each story
        # [23, 14,....]
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]


        y = np.zeros(len(word_index)+1)

        y[word_index[answer]]= 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

# PLACEHOLDER shape = (max_story, batch_size)
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))
# those are place holders ready to recieve input layer on of banches of stories and questions
vocab_size = len(vocab) + 1

#INPUT ENCODER M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.2))

# OUTPUT
# (samples,story_maxLen, embedding_dim)

#INPUT ENCODER C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim= max_question_len))
input_encoder_c.add(Dropout(0.2))

# OUTPUT
# (samples,story_maxLen, max_question_len)

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,output_dim= 64,input_length=max_question_len))
question_encoder.add(Dropout(0.2))

# (sample,query_maxlen,embedding_dim)

# ENCODED <------- ENCODER(INPUT)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation('softmax')(match)

response = add([match,input_encoded_c])
response = Permute((2,1))(response)

answer =  concatenate([response, question_encoded])
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#history = model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=120,validation_data=([inputs_test, queries_test], answers_test))
#filename = 'chatbot_120epochs.h5'
#model.save(filename)
model.load_weights('chatbot_120epochs.h5')

pred_results = model.predict(([inputs_test,queries_test]))
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])

my_story = str(input("Enter your story:"))
my_question = str(input("Enter your question:"))
answer = str(input("Enter the answer(yes or no):"))
if(answer != 'yes' and answer != 'no'):
    ptint("Invalid answer.")
    answer = 'no'

mydata = [(my_story.split(), my_question.split(), answer)]
my_story, my_ques, my_ans = vectorize_stories(mydata)
pred_results = model.predict(([my_story, my_ques]))
val_max= np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
