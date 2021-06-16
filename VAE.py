import argparse

parser = argparse.ArgumentParser(
    description='Script para ejecutar VAE',
)
parser.add_argument('-i',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-iv',help='Direccion texto de validacion de entrada' , action="store", required=True)
parser.add_argument('-o',help='Nombre de archivo de salida' , action="store", required=True)
parser.add_argument('-ov',help='Nombre de archivo de salida para texto de validacion' , action="store", required=True)
parser.add_argument('-l',help='longitud maxima de palabras' , action="store", required=True, type=int)
parser.add_argument('-w',help='Direccion indice de palabras' , action="store", required=True)

args = parser.parse_args()

INPUT_FILE = args.i
INPUT_FILE_VALIDATION = args.iv
OUTPUT_FILE = args.o
OUTPUT_FILE_VALIDATION = args.ov
MAX_LENGTH = int(args.l)
WORD_INDEX = args.w



from tensorflow.keras import losses, backend as K
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras
from tensorflow.keras import optimizers as KO
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
import json






os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
disable_eager_execution()


class VAE(object):
    def create(self, vocab_size=500, max_length=300, latent_rep_size=200):
        self.encoder = None
        self.decoder = None
        self.sentiment_predictor = None
        self.autoencoder = None 

        x = Input(shape=(max_length,127))
        x_embed = Embedding(vocab_size, 64, input_length=max_length)(x)
        
        vae_loss, encoded = self._build_encoder(x, latent_rep_size=latent_rep_size, max_length=max_length)
        self.encoder = Model(inputs=x, outputs=encoded)

        encoded_input = Input(shape=(latent_rep_size,))
        predicted_sentiment = self._build_sentiment_predictor(encoded_input)
        self.sentiment_predictor = Model(encoded_input, predicted_sentiment)

        decoded = self._build_decoder(encoded_input, vocab_size, max_length)
        self.decoder = Model(encoded_input, decoded)

        self.autoencoder = Model(inputs=x, outputs=self._build_decoder(encoded, vocab_size, max_length))
        opt = KO.Adam(lr=0.001)
        self.autoencoder.compile(optimizer=opt,
                                 loss=vae_loss,
                                 metrics=['accuracy'])
        

    def _build_encoder(self, x, latent_rep_size=200, max_length=300, epsilon_std=0.01):
        h = Bidirectional(LSTM(500, return_sequences=True, name='lstm_1'), merge_mode='concat')(x)
        h = Bidirectional(LSTM(500, return_sequences=False, name='lstm_2'), merge_mode='concat')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
        h = Dropout(0.35)(h)
        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            loss = tf.losses.CategoricalCrossentropy(from_logits=False)
            
            xent_loss = max_length * loss(x, x_decoded_mean)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _build_decoder(self, encoded, vocab_size, max_length):
        repeated_context = RepeatVector(max_length)(encoded)

        h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
        h = LSTM(500, return_sequences=True, name='dec_lstm_2')(h)
        h = Dropout(0.35)(h)
        decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)

        return decoded
    def _build_sentiment_predictor(self, encoded):
        h = Dense(100, activation='linear',name='sentiment_Predictor')(encoded)

        return Dense(1, activation='sigmoid', name='pred')(h)

index = imdb.get_word_index()


NUM_WORDS = 2000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS) #Tokenized representation


#HACER SPLIT DE SET DE VALIDACION
with open(INPUT_FILE, 'r') as filehandle:
    X_train = np.array(json.load(filehandle))


with open(INPUT_FILE_VALIDATION, 'r') as filehandle:
    X_test = np.array(json.load(filehandle))

with open(WORD_INDEX, 'r') as filehandle:
    index = json.load(filehandle)

NUM_WORDS = len(index)
print(index)

print("\n START \n")

print("Training data")
print(X_train.shape)
print(y_train.shape)
print(NUM_WORDS)
print("Number of words:")
print(len(np.unique(np.hstack(X_train))))

#Padding words

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post')
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post')



#Shuffle train data
train_indices = np.random.choice(np.arange(X_train.shape[0]), 153 ,replace=False)
test_indices = np.random.choice(np.arange(X_test.shape[0]), 153, replace=False)

X_train = X_train[train_indices]
y_train = y_train[train_indices]

X_test = X_test[test_indices]
y_test = y_test[test_indices]

#One-hot Encoding
temp = np.zeros((X_train.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_train.shape[0]), axis=0).reshape(X_train.shape[0], 1), np.repeat(np.array([np.arange(MAX_LENGTH)]), X_train.shape[0], axis=0), X_train] = 1

X_train_one_hot = temp

temp = np.zeros((X_test.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_test.shape[0]), axis=0).reshape(X_test.shape[0], 1), np.repeat(np.array([np.arange(MAX_LENGTH)]), X_test.shape[0], axis=0), X_test] = 1

x_test_one_hot = temp

key_list = list(index.keys())
val_list = list(index.values())

position = val_list.index(1)

print(X_train.shape)
print(X_train_one_hot.shape)
print(X_test.shape)
print(x_test_one_hot.shape)
def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + \
               model_name + "-{epoch:02d}-k.h5"
    directory = os.path.dirname(filepath)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=False)

    return checkpointer

def train():
    model = VAE()
    model.create(vocab_size=NUM_WORDS, max_length=MAX_LENGTH, latent_rep_size=2000)
    checkpointer = create_model_checkpoint('models', 'rnn_ae')
    model.autoencoder.summary()
    model.autoencoder.fit(x=X_train_one_hot, y={'decoded_mean': X_train_one_hot},
                          batch_size=2, epochs=1, callbacks=[checkpointer],
                          validation_data=(x_test_one_hot, {'decoded_mean': x_test_one_hot}))
    return model

model = train()
predictions = model.autoencoder.predict(X_train_one_hot)
prediction = predictions[0]
print(len(prediction))
print(len(prediction[0]))
print(len(prediction[1]))
outtrain = ''


for text in predictions:
    decodedSentence = ""
    
    for wordEncoded in text:
        result = np.where(wordEncoded == np.amax(wordEncoded))[0]
        #print(result)
        if(result != 0):
            position = val_list.index(result)
            wordDecoded = key_list[position]
            decodedSentence = decodedSentence + wordDecoded
    #print(len(text))
    outtrain=outtrain + decodedSentence + '\n'
    #print(text)
    #print(len(text[0]))
    #exit()

predictions = model.autoencoder.predict(x_test_one_hot)
prediction = predictions[0]
print(len(prediction))
print(len(prediction[0]))
print(len(prediction[1]))
outtest = ''


for text in predictions:
    decodedSentence = ""
    
    for wordEncoded in text:
        result = np.where(wordEncoded == np.amax(wordEncoded))[0]
        #print(result)
        if(result != 0):
            position = val_list.index(result)
            wordDecoded = key_list[position]
            decodedSentence = decodedSentence + wordDecoded
    #print(len(text))
    outtest=outtest + decodedSentence + '\n'
    #print(decodedSentence)
    #print(text)
    #print(len(text[0]))
    #exit()
    

file1 = open(OUTPUT_FILE, 'w')
file2 = open(OUTPUT_FILE_VALIDATION, 'w')

file1.write(outtrain)
file2.write(outtest)
file1.close()
file2.close()