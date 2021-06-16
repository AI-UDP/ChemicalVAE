# Glosario y descripcion de codigos

CHEMICALTEST.PY ... convierte archivo de cadenas quimicas a una representacion de indices
			("drugsOut.txt")

VAE.PY ... 	toma la representacion de indices a entrenar, otra de validacion y la 
		procesa en la vae, obtiene predicciones para los valores de prueba y 
		los de validacion ('listfileoriginal.txt','listfilevalidate.txt','wordindex.txt')

LABELING.PY... Este archivo toma como entrada el archivo original y el indice de palabras
		(Mismo que archivo que entra y sale en el primer script). 
		y como salida generara un archivo con la representacion decodificada con la etiqueta
		de cada quimico, un archivo con la representacion en indices y otro con las etiquetas 
		de cada farmaco listas para el archivo FFN.PY, asi con el primer archivo se consulta
		 

FFN.PY ... 	Esto Toma 2 pares de archivos, uno de entrenamiento y otro de prueba
		de los 2 pares de archivo uno debe ser la representacion de indices
		y el otro los labels de las representaciones (Deben estar mismas posiciones,
		Esto quiere decir que en la primera posicion del arreglo de la representacion
		de indices el label correspondiente a esa representacion debe igual estar
		en esa posicion).  Esto da como salida un archivo con la  matabilidad o no
		reproducibilidad del quimico

# Reproduccion de codigo
Comandos para replicar en orden

chemicalTest.py -i "drugsOut.txt" -o "listOutarg.txt" -wo "wodindexarg.txt"
VAE.py -i "listOutarg.txt" -iv "listOutarg.txt" -o "vaeout.txt" -ov "vaevalout.txt" -l "8" -w "wodindexarg.txt"
labeling.py -i "drugsOut.txt" -ofr "labelingoutarg.txt" -ol "labelinglabelout.txt" -w "wodindexarg.txt"
ffn.py -i "labelingoutarg.txt" -il "labelinglabelout.txt" -it "labelingoutarg.txt" -itl "labelinglabelout.txt" -o "fnnoutarg.txt" -id "vaeout.txt"

# Dependencias

Las dependencias se encuentran en el archivo requirements.txt