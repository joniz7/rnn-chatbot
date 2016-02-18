import os 
import random
import operator
import sys

from collections import Counter
execfile("parser.py")

def splitApostrophe(line):
  return line.replace("n't", " n't").replace("'s", " 's").replace("'re", " 're").replace("'ve", " 've").replace("'d", " 'd").replace("'ll", " 'll").replace("'m", " 'm")

trainInputFile = open("../data/train-data.utte", "w")
trainOutputFile = open("../data/train-data.resp", "w")

validInputFile = open("../data/valid-data.utte", "w")
validOutputFile = open("../data/valid-data.resp", "w")

trainInputFile.truncate()
trainOutputFile.truncate()
validInputFile.truncate()
validOutputFile.truncate()


usedMovies = []
lines = []


print "================== Parsing dialogue corpus ===================="

cnt = 0

os.chdir("../data/dialogs")
for root, dirs, files in os.walk(os.getcwd()):
	for f in [f for f in files if ".txt" in f]:
		if f not in usedMovies:
			usedMovies.append(f)
			lines = lines + parseFile(root+"/"+f)
			print cnt
			cnt += 1

print "================== Creating training and validation data ===================="

random.seed()
for i in range(len(lines)-1):
	utt, resp = (splitApostrophe(lines[i])+"\n", splitApostrophe(lines[i+1])+"\n")
	if random.randint(0,10) > 3:
		trainInputFile.write(utt)
		trainOutputFile.write(resp)
	else:
		validInputFile.write(utt)
		validOutputFile.write(resp)

print "Done."