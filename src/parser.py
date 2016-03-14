from collections import Counter
import codecs
import re
import operator

splitApostropheWords = ["n't", "'s", "'re", "'ve", "'d", "'ll", "'m"]
punctuationMarks = [",", ".", "!", "?", ":", ";"]

def splitApostrophe(line):
	global splitApostropheWords
	for w in splitApostropheWords:
		line = line.replace(w, " "+w)
	return line

def getTightJoinTokens(vocab):
	global splitApostropheWords
	global punctuationMarks
	return [vocab.get(w) for w in (splitApostropheWords + punctuationMarks)]


def containsAny(str, set):
    """ Check whether sequence str contains ANY of the items in set. """
    return 1 in [c in str for c in set]

def allCaps(str):
	return len([w for w in str if (w.isalpha() and w.isupper()) or not w.isalpha()]) == len(str)

def wordCount(filename):
	txt = open(filename)
	return len(txt.read().split())

# returs word -> wordCount dictionary
def countedWords(filename):
	txt = codecs.open(filename, encoding="utf-8")

	ws = Counter()

	"""for line in txt:
		for word in splitSentence(line):
			ws[word] += 1"""

	return Counter(txt.read().split())


def uniqueWords(filename):
	txt = open(filename)
	return set(w.lower() for w in txt.read().split())

"""
	for line in txt:
		splits = line.split()
		for word in splits:
			if word not in uniqueWords:
				uniqueWords.append(word)
				count = count + 1
"""

def splitSentence(s):
	words = []
	w = ""
	for c in s:
		if c.isalpha() or c.isdigit() or c == '-':
			w = w + c
		elif c == ' ' or c == '\t' or c == '\n':
			if w:
				words.append(w)
				w = ""
		else:
			if w:
				words.append(w)
				w = ""
			words.append(c)

	return words

def isCorrect(line):
	correct = True
	line = line.lower().split()
	if len(line) > 1:
		correct = not (line[0] == "cut" and (line[1] == "to:" or line[1] == "to"))
	correct = correct and reduce(operator.and_, [not c.isdigit() for c in line[0]], True)
	return correct

def removeStars(line):
	return re.sub("\.(\.+)", " _DOTS ", line)

def purgeLine(line):
	return removeStars(re.sub("\(.*?\)", "", line.replace("\n", " ").lower())).replace("*", "")

def parseFile(filename):
	txt = open(filename)
	#targetFile = open("data", "w")
	#targetFile.truncate()

	lines = []

	#print containsAny("<b><!--", ["<", ">","/","\\"])

	totalLine = ""

	for line in txt:
		if not line == "\n":
			if not containsAny(line, ["<", ">","/","\\", "=", "--"]) and isCorrect(line):
				line = re.sub("\(.*?\)", "", line.replace("\n", ""))
				if allCaps(line.replace(" ", "")) and totalLine:
					#print "%s     %d"%(line, len(line.strip()))
					#wordSplit = splitSentence(totalLine)
					#words.append((oldLine, wordSplit))
					lines.append(purgeLine(totalLine))
					totalLine = ""
				elif allCaps(line.replace(" ", "")):
					#print " elif %s     %d"%(line, len(line.strip()))
					totalLine = ""
				else:
					#print " else %s     %d"%(line, len(line.strip()))
					totalLine += " "+line
	if totalLine:
		lines.append(purgeLine(totalLine))
	return lines

def parseEmbeddings(filename):
	file = open(filename)
	mat = []	
	for line in file:
		mat.append(map(float, line.split()[1:]))

	return mat