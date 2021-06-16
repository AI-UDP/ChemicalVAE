import numpy as np
import json
import re

import argparse

parser = argparse.ArgumentParser(
    description='Script para ejecutar VAE',
)
parser.add_argument('-i',help='Direccion texto entrada' , action="store", required=True)
parser.add_argument('-ofr',help='Nombre de archivo de salida' , action="store", required=True)
parser.add_argument('-ol',help='Nombre de archivo de salida para texto de validacion' , action="store", required=True)
parser.add_argument('-w',help='Direccion indice de palabras' , action="store", required=True)

args = parser.parse_args()

INPUT_FILE = args.i
OUTPUT_FILE_REP = args.ofr
OUTPUT_FILE_LABEL = args.ol
WORD_INDEX = args.w


def separateElements(s):
    sub = []
    final = []
    char = ""
    num = ""
    for letter in s:
        if letter.isdigit():
            if char:
                out = re.findall('[A-Z][^A-Z]*', char)
                final = final + out
                sub.append(char)
                char = ""
            num += letter
        else:
            if num:
                sub.append(num)
                num = ""
            char += letter
    if char:
        
        out = re.findall('[A-Z][^A-Z]*', char)
        final = final + out
        sub.append(char)
        sub = sub + out
    else:
        sub.append(num)
    if(len(sub[0])>2):
        sub.pop(0)
    if(sub[0]=='NN'):
        sub.pop(0)
        sub.insert(0,'N')
        sub.insert(0,'N')
    if 'OO' in sub:
        index = sub.index('OO')
        sub.pop(index)
        sub.insert(index,'O')
        sub.insert(index,'O')
    if 'HH' in sub:
        index = sub.index('HH')
        sub.pop(index)
        sub.insert(index,'H')
        sub.insert(index,'H')  
    print(sub)
    return sub










with open(WORD_INDEX) as json_file:
    wordIndex = json.load(json_file)
outLine = open(INPUT_FILE,'r')


lines = outLine.readlines()
out = []
outTrain = []
for i in lines:
    word = separateElements(i.split(':')[0]) 
    print(word)
    append = []
    for j in word:
        try:
            append.append(wordIndex[j])
        except KeyError:
            val=0
    outTrain.append((i.split(':')[1]).replace('\n',''))
    out.append(append)

with open(OUTPUT_FILE_REP, 'w') as filehandle:
    json.dump(out, filehandle)
with open(OUTPUT_FILE_LABEL, 'w') as filehandle:
    json.dump(outTrain, filehandle)

