from mendeleev import element
import re
import random
import pickle
import json
import argparse

parser = argparse.ArgumentParser(
    description='Script obtener representacion de indices e indice de palabras',
)
parser.add_argument('-i',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-o',help='Nombre texto salida' , action="store", required=True)
parser.add_argument('-wo',help='Nombre de salida indice' , action="store", required=True)
args = parser.parse_args()

INPUT_FILE = args.i
OUTPUT_FILE = args.o
OUTPUT_INDEX = args.wo

elements = ["C10H12P2O5S;0","C10N12N2O5S;0"]

archivo = open(INPUT_FILE,'r')
elements = []

lines = archivo.readlines()

for i in lines:
    line = i.replace('\n','')
    elements.append(line)

archivo.close()
def separateElements(s):
    sub = []
    char = ""
    num = ""
    for letter in s:
        if letter.isdigit():
            if char:
                sub.append(char)
                char = ""
            num += letter
        else:
            if num:
                sub.append(num)
                num = ""
            char += letter
    sub.append(char) if char else sub.append(num)
    return sub

def indexW(string, wordList):
    
    indexDict = {" ": 0}
    res = []
    count = 1
    for line in string:
        for word in wordList:
            if word in line:
                indexDict[word] = wordList.index(word)
    return indexDict

def createWordList(phrase, listr):

    for word in phrase:
        if(word not in listr):
            listr.append(word)
    return listr

wordList = [" "]

for i in elements:
    word = separateElements(i.split(':')[0])
    wordList = createWordList(word,wordList)

wordIndex = []
wordIndex = indexW(elements,wordList)

#print(wordList)

#print(wordIndex)


no_integers = [x for x in wordList if not (x.isdigit() 
                                         or x[0] == '-' and x[1:].isdigit())]
integers = [x for x in wordList if (x.isdigit() 
                                         or x[0] == '-' and x[1:].isdigit())]
numbers_list = range(0,80)
typeList = [0,1]

no_integers.remove(' ')
count = 0
limit = 10000



generatedFileRead = open(INPUT_FILE,"r")

lines = generatedFileRead.readlines()
out = []
for i in lines:
    word = separateElements((i.split(':')[0]).replace('\n','')) 
    append = []
    for j in word:
        append.append(wordIndex[j])
    out.append(append)



with open(OUTPUT_FILE, 'w') as filehandle:
    json.dump(out, filehandle)

with open(OUTPUT_INDEX, 'w') as filehandle:
    json.dump(wordIndex, filehandle)

