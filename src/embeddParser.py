import sys
import random
execfile("statistics.py")

def printValues(values):
  res = ""
  for v in values:
    res += str(v)+" "
  return res

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def unkValues(dim, means, variances):
  return [random.gauss(means[i], variances[i] ** 0.5) for i, _ in enumerate(range(dim))]

if(len(sys.argv) < 3):
  print "Not enough arguments"
  sys.exit()

vocabFile = open(sys.argv[1])
gloveFile = open(sys.argv[2])
newEmbed = open("../data/embeddings.txt", "w")

vocab = {}

rowToWord = []

print "================ read vocab ======================"

for i,line in enumerate(vocabFile):
  vocab[line.strip()] = {'row': i}
  rowToWord.append(line.strip())
  if(i % 1000 == 0):
    print str(i)+"\n"

print "vocab size: "+str(len(vocab))

print "================ read embedd ======================"

i = 0

for line in gloveFile:
  values = line.split()
  if values[0] in vocab:
    vocab[values[0].lower()]['embedding'] = values[1:]
  i += 1
  if(i % 5000 == 0):
    print str(i)

embeddings = []
for word in vocab:
  if 'embedding' in vocab[word]:
    embeddings.append(vocab[word]['embedding'])

means = mean(map(list, zip(*embeddings)))
variances = variance(map(list, zip(*embeddings)))

print "================ write file ======================"

notInEmbed = 0

nem = open("../data/notInEmbed.txt", "w")

for i, word in enumerate(rowToWord):
  vocWord = vocab[word]
  if('embedding' in vocWord):
    newEmbed.write(word+" "+printValues(vocWord['embedding'])+"\n")
  else:
    newEmbed.write(word+" "+printValues(unkValues(int(sys.argv[3]), means, variances))+"\n")
    nem.write(word+"\n")
    notInEmbed += 1
  #if(i%1000 == 0):
    #print str(i)

print "Number of words not in embedd: "+str(notInEmbed)