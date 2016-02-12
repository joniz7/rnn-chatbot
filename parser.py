from collections import Counter
import codecs


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


def parseFile(filename):
	txt = open(filename)
	#targetFile = open("data", "w")
	#targetFile.truncate()

	lines = [[],[]]

	#print containsAny("<b><!--", ["<", ">","/","\\"])

	totalLine = ""
	oldLine = "GO"

	for line in txt:
		if not line == "\n":
			if not containsAny(line, ["<", ">","/","\\", "=", "--"]):
				if allCaps(line) and totalLine:
					#wordSplit = splitSentence(totalLine)
					#words.append((oldLine, wordSplit))
					lines[0].append(oldLine.replace("\n", ""))
					lines[1].append(totalLine.replace("\n", ""))
					oldLine = totalLine
					totalLine = ""
				else:
					totalLine += line
	return lines