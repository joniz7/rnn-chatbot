import os 
import codecs
import random
from collections import Counter

execfile("parser.py")

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
			movieLines = parseFile(root+"\\"+f)
			lines = lines + movieLines
			#words = words.union(uniqueWords(root+"\\"+f))
			#wordCnt = wordCnt + wordCount(root+"\\"+f)
			#cWords += countedWords(root+"\\"+f)
#			movies.append((f, parseFile(root+"\\"+f)))
#			monster.write(open(root+"\\"+f).read())


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
	if random.randint(0,10) > 3:
		trainInputFile.write(lines[i]+"\n")
		trainOutputFile.write(lines[i+1]+"\n")
	else:
		validInputFile.write(lines[i]+"\n")
		validOutputFile.write(lines[i+1]+"\n")