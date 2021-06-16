
import argparse

parser = argparse.ArgumentParser(
    description='Script para ejecutar VAE',
)
parser.add_argument('-i',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-il',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-it',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-itl',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-o',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-id',help='Direccion texto entrada' , action="store", required=True)



args = parser.parse_args()

INPUT_FILE = args.i
INPUT_FILE_LABEL = args.il
INPUT_FILE_TEST = args.it
INPUT_FILE_TEST_LABEL = args.itl
OUTPUT_FILE = args.o
INPUT_FILE_DECODED = args.id


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json




with open(INPUT_FILE) as json_file:
    drugsX = json.load(json_file)
with open(INPUT_FILE_LABEL) as json_file:
    drugsY = json.load(json_file)
with open(INPUT_FILE_TEST) as json_file:
    drugsXTest = json.load(json_file)
with open(INPUT_FILE_TEST_LABEL) as json_file:
    drugsYTest = json.load(json_file)

if(len(max(drugsX,key=len))>len(max(drugsXTest,key=len))):
    MAX_LENGTH = len(max(drugsX,key=len))
else:
    MAX_LENGTH = len(max(drugsXTest,key=len))


drugsX = pad_sequences(drugsX, maxlen=MAX_LENGTH, padding='post')
drugsXTest = pad_sequences(drugsXTest, maxlen=MAX_LENGTH, padding='post')
lb = LabelBinarizer()
drugsYbinary = lb.fit_transform(drugsY)
drugsYTestBinary = lb.fit_transform(drugsYTest)


drugsX = drugsX.astype("float32") / 126.0
drugsXTest = drugsXTest.astype("float32") / 126.0

print(drugsXTest.shape)
print(drugsYTestBinary.shape)


model = Sequential()

model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


print("[INFO] training network...")
sgd = Adam(0.001)

model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
    
H = model.fit(drugsX, drugsYbinary, validation_data=(drugsX, drugsYTestBinary),
	epochs=100, batch_size=5)


print("[INFO] evaluating network...")
predictions = model.predict(drugsXTest, batch_size=5)
archivout = open(OUTPUT_FILE,'w')
archivred = open(INPUT_FILE_DECODED,'r')
archivred = archivred.readlines()
stop= len(predictions)
count = 0
archivredProc = []

for line in archivred:
    line = line.replace('\n','')
    archivredProc.append((line.split(":"))[0])

while count < stop:
    outlineappend = archivredProc[count] + ":" + str(predictions[count][0]) + '\n'
    archivout.write(outlineappend)
    count = count + 1

archivout.close()


