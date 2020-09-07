# encoding = utf-8

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Lambda, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Embedding, Dropout, LSTM, Conv1D, \
    Bidirectional, MaxPooling1D
from keras import backend as K
from keras.models import Sequential
import nltk
from nltk.stem import WordNetLemmatizer
import pandas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping
import emoji
import string as str
import demoji
demoji.download_codes()
from nltk.tag import CRFTagger
from nltk import pos_tag
import pycrfsuite
from collections import defaultdict
from nltk.corpus import wordnet as wn
import re
from sklearn.preprocessing import OneHotEncoder
from keras.layers import LeakyReLU
from keras import regularizers
from keras.utils.vis_utils import plot_model
np.random.seed(500)

# Defining lemmatizer
lemmatizer = WordNetLemmatizer()


# - Remove puntuation method
def remove_PunctuationAndNum(Word):
    import string as str

    # define punctuation
    punctuation = """:#'’!'!"#$%&()”+,-./\:;<=>?[\\]“^_`{|}~\t\n'"""
    output_word = ''
    for char in Word:
        if char not in punctuation and not char.isnumeric():
            output_word = output_word + char
    return output_word


# Data Preprocessing

train_df = pandas.read_csv('Data_Set/Dataset_updated.csv', encoding='utf-8')
train_df = train_df.sample(frac=1).reset_index(drop=True)
X_train = train_df['tweet']
Y_train = train_df['Label']

tokenizer = nltk.tokenize.TreebankWordTokenizer()
TAGGER_PATH = "crfpostagger"
POS_tagger = CRFTagger()  # initialize tagger
POS_tagger.set_model_file(TAGGER_PATH)

# Dictionary for apostrophe characters
apost_Dict = {
    "m": "am",
    "n't": "not",
    "'s": "is",
    "re": "are",
    "ve": "have",
    "ll": "will",
    "d": "did",
    "cause": "because",
    "c'mon": "come on",
    "nt": "not",
    "s": "is",
    "t": "not"
}

# - Dictionary for short hands - #
short_word_dict = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace",
"cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend",
"gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked",
"irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female",
"m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if",
"wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
}

# - Short hand convert function - #
def Convert_Short_Hands(words):
    sent_temp = []
    for word in words:
        if word in short_word_dict:
            word = short_word_dict[word]
            if word.find(' ') > -1:

                for val in word.split():
                    sent_temp.append(val)
            else:
                sent_temp.append(word)

        else:
            sent_temp.append(word)

    return sent_temp

# dictionary defination for POS tag
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

X_train_preprocessed = []
print('\n\nPreprocessing The Data .......\n\n')
for sent in X_train:

    sent_processed = []
    sent = emoji.demojize(sent)
    sent = sent.lower()
    sent = re.sub(r"http\S+", "", sent)
    sent = re.sub(r"@\S+", "", sent)
    sent = sent.replace('url', '')
    words = tokenizer.tokenize(sent)

    words = Convert_Short_Hands(words)

    for word, tag in POS_tagger.tag(words):

        if word in apost_Dict:
            word = apost_Dict[word]
        word = remove_PunctuationAndNum(word)
        word = word.lower()

        if word != "":
            word = lemmatizer.lemmatize(word, tag_map[tag[0]])
            sent_processed.append(word)

    X_train_preprocessed.append(sent_processed)

# print(X_train_preprocessed)

t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(X_train_preprocessed)
word2idx = t.word_index
word2idx = {k: (v + 2) for k, v in word2idx.items()}
word2idx["<PAD>"] = 0
word2idx["<START>"] = 1
word2idx["<UNK>"] = 2

################################################################################
# - Readying Inputs - #
# print(word2idx)
from keras.preprocessing.sequence import pad_sequences

MAXIMUM_LENGTH = 200
# print(Y_train_a)
Label_ENC_dict = {
    'norm': [0,0,1],
    'Abuse/harassment': [0,1,0],
    'Hateful_conduct': [1,0,0]
}

# Label_Decode_Dic =  {
#         [0,0,1] :'norm',
#         [0,1,0] : 'Abuse/harassment',
#         [1,0,0] : 'Hateful_conduct'
#     }
# Encoding
X_train_encoded = []
for sent in X_train_preprocessed:
    sentENC = []
    for word in sent:
        sentENC.append(word2idx[word])
    X_train_encoded.append(sentENC)



Y_train_ENC = []

for value in Y_train:
    # print(value)
    Y_train_ENC.append(Label_ENC_dict[value])

X_train_preprocessed_final = pad_sequences(X_train_encoded, maxlen=MAXIMUM_LENGTH)
# preprocessed_test_data = pad_sequences(test_data, maxlen=MAXIMUM_LENGTH)
# print(X_train_preprocessed_final)

# print(Y_train_a_ENC)

################################################################################
VOCAB_SIZE = len(word2idx)

EMBED_SIZE = 200

def F1_Custom_Loss(y_true,y_pred):
        y_true  = K.cast(y_true,'float')
        y_pred = K.cast(y_pred,'float')

        ground_positives = K.sum(y_true, axis=0)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
        #return 1 - K.mean(f1)
        return 1 - K.sum(weighted_f1)

# - Defining optimizer with learning rate - #

optimizer = keras.optimizers.Adam(lr=0.001)

# Model LSTM
model1 = Sequential()
model1.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=MAXIMUM_LENGTH, name='Embedding_layer'))
model1.add(Dropout(rate=0.5, noise_shape=None, seed=None))
model1.add(LSTM(units=100, return_sequences=True, activation='tanh', name='LSTM_layer_1'))
model1.add(Dropout(rate=0.5, noise_shape=None, seed=None))
model1.add(LSTM(units=100, activation='tanh', name='LSTM_layer_2'))
model1.add(Dropout(rate=0.5, noise_shape=None, seed=None))
model1.add(Dense(units=3, activation='softmax', name='Output_layer'))
model1.summary()

model1.compile(
    loss='categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])

###################################################################################
# - Bi-LSTM - #
# model2 = Sequential()
# model2.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=MAXIMUM_LENGTH, name='Embedding_layer'))
# model2.add(Dropout(rate=0.5, noise_shape=None, seed=None))
# model2.add(
#     Bidirectional(LSTM(units=100, return_sequences=True, activation='tanh', name='LSTM_layer_1'), merge_mode='concat'))
# # model2.add(Dropout(rate = 0.5, noise_shape=None, seed=None))
# # model2.add(LSTM(units = 100,activation='tanh',name ='LSTM_layer_2'))
# model2.add(Dropout(rate=0.5, noise_shape=None, seed=None))
# model2.add(Dense(units=3, activation='softmax', name='Output_layer'))
# model2.summary()

# model2.compile(
#     loss='categorical_crossentropy', optimizer='adam',
#     metrics=['accuracy'])

####################################################################################
# - CNN Model - #
model3 = Sequential()
model3.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=MAXIMUM_LENGTH, name='Embedding_layer'))
model3.add(Conv1D(128, kernel_size=7, activation='relu', padding='valid'))
model3.add(GlobalMaxPooling1D())
model3.add(Dense(100,activation='relu'))#,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
#model3.add(LeakyReLU(alpha=0.07))
model3.add(Dense(30, activation = 'relu'))
model3.add(Dense(3, activation='sigmoid'))
model3.summary()

model3.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
Model_CNN = model3.to_json()
################################################################################
# - CNN LSTM model - #
model4 = Sequential()
model4.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=MAXIMUM_LENGTH, name='Embedding_layer'))
model4.add(Conv1D(128, kernel_size=7, activation='relu', padding='valid'))
model4.add(MaxPooling1D())  # pool_size=2
# model3.add(Dense(200, activation='relu'))
model4.add(Dropout(rate=0.3, noise_shape=None, seed=None))
model4.add(LSTM(units=200, name='LSTM_layer_1'))
model4.add(LeakyReLU(alpha=0.07))
model4.add(Dropout(rate=0.3, noise_shape=None, seed=None))
model4.add(Dense(30,activation = 'relu'))
model4.add(Dense(3, activation='softmax'))
model4.summary()

model4.compile(optimizer=optimizer,
               loss='categorical_crossentropy',
               metrics=['accuracy'])


Model_CNN_LSTM = model4.to_json()

# One Hot encoding the labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json,load_model
label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(Y_train_M)
#
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_train_Y = onehot_encoder.fit_transform(integer_encoded)

# print(onehot_train_Y)
# print('unique = ',Y_train_M.unique())
# print('int unq =', integer_encoded.unique())


################################################################################
# - Train Val Test split - #
Train_Samples = 25000

partial_X_train = np.array(X_train_preprocessed_final[:Train_Samples])
X_heldout = np.array(X_train_preprocessed_final[Train_Samples:])

partial_y_train= np.array(Y_train_ENC[:Train_Samples])
y_heldout = np.array(Y_train_ENC[Train_Samples:])

Val_Size = 4000
X_test_try = np.array(X_heldout[Val_Size:])
Y_test_try = np.array(y_heldout[Val_Size:])

X_val = np.array(X_heldout[:Val_Size])
y_val = np.array(y_heldout[:Val_Size])


# print('y =',y_val)
# print('len =', len(y_val))
# print('y_enc = ', partial_y_train)
###########################################################################
# - Training - #
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

Best_accuracy = 0
print('\n\nTraining the model .....\n\n')
import time
for i in range(5):
    model = model_from_json(Model_CNN)
    model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    history = model.fit(partial_X_train,
                         partial_y_train,
                         epochs=1,
                         batch_size=100,
                         validation_data=(X_val, y_val),
                         verbose=1, callbacks=[es])

    #plot_model(model3, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    #print(history.history['val_accuracy'][-2])
    Acc = history.history['val_accuracy'][-1]


    if Acc > Best_accuracy:
        #time.sleep(100)
        Best_accuracy = Acc
        # results = model3.evaluate(X_val, y_val, batch_size=100, verbose=1)
        # print(results)
        print('saving the model ...')
        model.save('CNN')
#print(Best_accuracy)
Best_Model = load_model('CNN')

# results = Best_Model.evaluate(X_val, y_val, batch_size= 100, verbose = 1)
# print(results)

##################################################################################

# history = model3.fit(partial_X_train,
#                          partial_y_train,
#                          epochs=1,
#                          batch_size=100,
#                          validation_data=(X_val, y_val),
#                          verbose=1, callbacks=[es])
#
# results = model3.evaluate(X_val, y_val, batch_size= 100, verbose = 1)
# print(results)
#####################################################################################
# - Testing - #

# test_df = pandas.read_csv('Data_Set/Test_Dataset.csv', encoding='utf-8')
# X_test = test_df['tweet']
# Y_test = test_df['labels']

# X_test_preprocessed = []
# for sent in X_test:

#     sent_processed = []
#     sent = emoji.demojize(sent)
#     sent = sent.lower()
#     sent = re.sub(r"http\S+", "", sent)
#     sent = re.sub(r"@\S+", "", sent)
#     sent = sent.replace('url', '')
#     words = tokenizer.tokenize(sent)

#     words = Convert_Short_Hands(words)
#     #print(words)
#     for word, tag in POS_tagger.tag(words):

#         if word in apost_Dict:
#             word = apost_Dict[word]
#         word = remove_PunctuationAndNum(word)
#         word = word.lower()

#         if word != "":
#             word = lemmatizer.lemmatize(word, tag_map[tag[0]])
#             sent_processed.append(word)

#     X_test_preprocessed.append(sent_processed)



# # - Encoding - #
# X_test_encoded = []
# for sent in X_test_preprocessed:
#   sentENC =[]
#   for word in sent:
#     if word not in word2idx:
#       word = '<UNK>'
#     sentENC.append(word2idx[word])
#   X_test_encoded.append(sentENC)

# X_test_encoded = pad_sequences(X_test_encoded, maxlen=MAXIMUM_LENGTH)

# X_test_encoded = np.array(X_test_encoded)
# Y_test_ENC = []
# for value in Y_test:
#     # print(value)
#     Y_test_ENC.append(Label_ENC_dict[value])

#######################################################################

# - Evaluating on TestData - #
val_pred = Best_Model.predict_classes(x= X_test_try, verbose=1,batch_size=100)#X_test_encoded, verbose=1)

#print(val_pred)
Labels = []
for lab in Y_test_try:#Y_test_ENC:
    Labels.append(np.argmax(lab))

print('\n\n\n')
print('################ - Final Results - ##################\n')
print("confusion_matrix for test data : \n", confusion_matrix(Labels, val_pred))
print ('F1 Score = ', f1_score(Labels,val_pred,average='weighted'))
print('Classification report =\n ', classification_report(Labels,val_pred, target_names= ['norm','Abuse/harassment','Hateful_conduct']))