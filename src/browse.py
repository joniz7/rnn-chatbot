import os 
import random
import operator
import sys

from collections import Counter
from operator import itemgetter

# percentage of partitions between training, validation and test data respectively
if(len(sys.argv) < 4):
  print "Not enough arguments"
  print "Need to give, in order:"
  print "Train data percantage"
  print "Validation data percentage"
  print "Test data percentage"
  print "Generate corpus information? (True/False) (Optional)"
  print "Split each sentence? (True/False) (Optional)"
  sys.exit()

train_data_percentage = int(sys.argv[1])
valid_data_percentage = int(sys.argv[2])
test_data_percentage = int(sys.argv[3])
generateCorpusInfo = False if len(sys.argv) <= 4 else bool(sys.argv[4])
split_sentence = False if len(sys.argv) <= 5 else bool(sys.argv[5])

execfile("parser.py")

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
			lines = lines + parseFile(root+"/"+f, splitSentence = split_sentence)
			if cnt % 50 == 0:
				print "Parsing script %d" % cnt
			cnt += 1

print "================== Creating training and validation data ===================="

numOfLongLines = 0
longest = []

# So the distribution will be deterministic
random.seed(1234567890)
totalProb = train_data_percentage + valid_data_percentage + test_data_percentage
for i in range(len(lines)-1):
	utt, resp = (splitApostrophe(lines[i])+"\n", splitApostrophe(lines[i+1])+"\n")
	if len(utt.split()) > 40 and generateCorpusInfo:
		longest.append((i, len(utt.split())))
		numOfLongLines += 1
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

if generateCorpusInfo:
	longest.sort(key=itemgetter(1))

	longFile = open("../data/longest.txt", "w")
	longFile.truncate()
	longnumFile = open("../data/longnum.txt", "w")
	longnumFile.truncate()

	lengthFile = open("../data/lengths.txt", "w")
	lengthFile.truncate()

	longFile.write(lines[longest[-1][0]])
	for _, l in longest:
		longnumFile.write("%d\n"%l)

	longCount = Counter([l for (_, l) in longest])
	for length in longCount:
		lengthFile.write("%d %d\n"%(length, longCount[length]))
	print "Number of long lines: %d"%numOfLongLines
	print "Longest line: %d"%longest[-1][1]
	print "Total lines %d"%len(lines)

print "Done."

os.chdir("../../src")

trainInputFile.close()
trainOutputFile.close()
validInputFile.close()
validOutputFile.close()
testInputFile.close()
testOutputFile.close()

