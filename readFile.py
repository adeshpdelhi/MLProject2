import numpy as np
from random import shuffle
def readDataSet(file):
	try:
		f = open(file, 'r')
	except IOError:
		print "UNABLE TO OPEN FILE\n"
		return (np.array([]),0,0)
	X = []
	for line in f:
		row = line.split(';')
		if(len(row) <= 1):
			continue
		row[-1] = row[-1].split('\n')[0]
		row = map(float, row[:])
		X.append(row)
	f.close()
	shuffle(X)
	# print (X[0:10])
	return np.array(X), len(X), len(X[0])