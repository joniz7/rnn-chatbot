import os 
import codecs
import random
from collections import Counter

execfile("parser.py")

def splitApostrophe(line):
  return line.replace("n't", " n't").replace("'s", " 's").replace("'re", " 're").replace("'ve", " 've").replace("'d", " 'd").replace("'ll", " 'll").replace("'m", " 'm")

#monster = codecs.open("data", "w")
#monster.truncate()

#targetFile = open("processed_data", "w")
#targetFile.truncate()

trainInputFile = open("..\\data\\train-data.utte", "w")
trainOutputFile = open("..\\data\\train-data.resp", "w")

validInputFile = open("..\\data\\valid-data.utte", "w")
validOutputFile = open("..\\data\\valid-data.resp", "w")

#wordListFile = codecs.open("wordCount", "w", encoding="utf-8")
#wordListFile.truncate()

movies = []

usedMovies = []

cnt = 0

words = set()

cWords = Counter()

wordCnt = 0

lines = []

for root, dirs, files in os.walk(os.getcwd()):
	for f in [f for f in files if ".txt" in f]:
		if f not in usedMovies:
			cnt = cnt+1
			print cnt
			usedMovies.append(f)
			lines = lines + parseFile(root+"\\"+f)
			#words = words.union(uniqueWords(root+"\\"+f))
			#wordCnt = wordCnt + wordCount(root+"\\"+f)
			#cWords += countedWords(root+"\\"+f)
			#movies.append((f, parseFile(root+"\\"+f)))
			#monster.write(open(root+"\\"+f).read())


#cWords = countedWords("data")


#print "Total number of words: %i"%wordCnt
#print "Unique number of words: %i"%len(words)
#print "10 Most common words:"
#print cWords.most_common(10)


"""  For creating vocabulary with word count
wordListFile.write("WORD \t\t\t\t\tCOUNT\n")
for w, c in cWords.most_common():
	wordListFile.write(str(w)+"\t\t\t\t\t%i"%c)
	wordListFile.write("\n")
"""
random.seed()
for i in range(len(lines)-1):
	utt, resp = (splitApostrophe(lines[i])+"\n", splitApostrophe(lines[i+1])+"\n")
	if random.randint(0,10) > 3:
		trainInputFile.write(utt)
		trainOutputFile.write(resp)
	else:
		validInputFile.write(utt)
		validOutputFile.write(resp)
