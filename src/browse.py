import os 
import random
import operator
import sys
import time

from collections import Counter

# 50/25/25 (of 100) partition between training, validation and test data respectively
train_data_size = 50
valid_data_size = 25
test_data_size = 25

execfile("parser.py")

def splitApostrophe(line):
  return line.replace("n't", " n't").replace("'s", " 's").replace("'re", " 're").replace("'ve", " 've").replace("'d", " 'd").replace("'ll", " 'll").replace("'m", " 'm")

trainInputFile = open("../data/train-data.utte", "w")
trainOutputFile = open("../data/train-data.resp", "w")

validInputFile = open("../data/valid-data.utte", "w")
validOutputFile = open("../data/valid-data.resp", "w")

testInputFile = open("../data/test-data.utte", "w")
testOutputFile = open("../data/test-data.resp", "w")

trainInputFile.truncate()
trainOutputFile.truncate()
validInputFile.truncate()
validOutputFile.truncate()
testInputFile.truncate()
testOutputFile.truncate()


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

random.seed(1234567890)
totalProb = train_data_size + valid_data_size + test_data_size
start_time = time.time()
for i in range(len(lines)-1):
	utt, resp = (splitApostrophe(lines[i])+"\n", splitApostrophe(lines[i+1])+"\n")
	ran = random.randint(0,totalProb)
	if train_data_size > ran:
		trainInputFile.write(utt)
		trainOutputFile.write(resp)
	elif train_data_size + valid_data_size > ran:
		validInputFile.write(utt)
		validOutputFile.write(resp)
	else:
		testInputFile.write(utt)
		testOutputFile.write(resp)
end_time = time.time()
print "Done."
print("Elapsed time was %g seconds" % (end_time - start_time))