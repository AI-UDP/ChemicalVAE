# Glosario y descripcion de codigos

CHEMICALTEST.PY ... 	converts chemical sequence file to an index representation
			("drugsOut.txt")

VAE.PY ... 	uses the index representation for training, a validation file and processes them in a VAE. 
		Generates predictions for test and validation values.
		('listfileoriginal.txt','listfilevalidate.txt','wordindex.txt')

LABELING.PY... 	This file takes the original input file and the word index both as input and
		generates a file with the decoded representation and label of each chemical compound,
		a file with the index representation and another one with the labels of each drug
		readily accessible for processing by the FFN.PY file.

FFN.PY ... 	This file takes two pairs of files: a training one and a test one. This
		file requires two pairs of input files - one with the index representation
		and the other one with the labels (at corresponding positions). This file
		outputs the value that directs whether the drug tends to kill bacteria or
		to slow down their growth.

# Reproduccion de codigo
The command sequence to run these code files is:

chemicalTest.py -i "drugsOut.txt" -o "listOutarg.txt" -wo "wodindexarg.txt"
VAE.py -i "listOutarg.txt" -iv "listOutarg.txt" -o "vaeout.txt" -ov "vaevalout.txt" -l "8" -w "wodindexarg.txt"
labeling.py -i "drugsOut.txt" -ofr "labelingoutarg.txt" -ol "labelinglabelout.txt" -w "wodindexarg.txt"
ffn.py -i "labelingoutarg.txt" -il "labelinglabelout.txt" -it "labelingoutarg.txt" -itl "labelinglabelout.txt" -o "fnnoutarg.txt" -id "vaeout.txt"

# Dependencias

Dependencies can be found in the requirements.txt file
