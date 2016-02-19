import os 
import random
import operator
import sys

from collections import Counter

# percentage of partitions between training, validation and test data respectively
if(len(sys.argv) < 4):
  print "Not enough arguments"
  sys.exit()

train_data_percentage = sys.argv[1]
valid_data_percentage = sys.argv[2]
test_data_percentage = sys.argv[3]

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
			if cnt % 50 == 0:
				print "Parsing script %d" % cnt
			cnt += 1

print "================== Creating training and validation data ===================="

random.seed(1234567890)
totalProb = train_data_percentage + valid_data_percentage + test_data_percentage
for i in range(len(lines)-1):
	utt, resp = (splitApostrophe(lines[i])+"\n", splitApostrophe(lines[i+1])+"\n")
	ran = random.randint(0,totalProb)
	if train_data_percentage > ran:
		trainInputFile.write(utt)
		trainOutputFile.write(resp)
	elif train_data_percentage + valid_data_percentage > ran:
		validInputFile.write(utt)
		validOutputFile.write(resp)
	else:
		testInputFile.write(utt)
		testOutputFile.write(resp)

print "Done."