"""Before running make sure the code is in root/src and the data is in root/data_path

    The dialog-corpus must be in a subfolder in data called dialogs.
    Before running make sure you at least have this:

    root:
      data/
      src/
      Dockerfile
      README.md

    src:
      all python files

    data
      dialogs/
      glove.6B.XXd.txt

    The dialogs and glove-vector are needed, the rest will be generated.

"""

import data_utils
import operator
import os
import sys

# Parameters
vocab_size = 300
embedding_dimension = 50

# percentage of partitions between training, validation and test data respectively
train_data_percentage = 50
valid_data_percentage = 25
test_data_percentage = 25

embeddings_filename = "embeddings"+str(vocab_size)+".txt"
data_path = "../data"

trainingFiles = [data_path+"/train-data.utte", data_path+"/train-data.resp", data_path+"/valid-data.utte", data_path+"/valid-data.resp"]

print "================== Checking if training and validation data exists ===================="
if(reduce(operator.and_, map(os.path.isfile, trainingFiles))):
  print "Exists, moving on"
else:
  sys.argv = ["browse.py", train_data_percentage, valid_data_percentage, test_data_percentage]
  execfile("browse.py")


vocabFiles = [data_path+"/vocab"+str(vocab_size), data_path+"/vocab"+str(vocab_size)+".utte"]
print "================== Checking if vocabulary exists ===================="
raw_input()
if(reduce(operator.and_, map(os.path.isfile, vocabFiles))):
  print "Exists, moving on"
else:
  data_utils.prepare_dialogue_data(data_path, vocab_size)

gloveFile = data_path+"/glove.6B."+str(embedding_dimension)+"d.txt"

print "================== Checking if embeddings exists ===================="
if not os.path.isfile(gloveFile):
  print "Found no glove vectors in the data folder"
else:
  if(os.path.isfile(data_path+"/"+embeddings_filename)):
    print "Embeddings exists"
  else:
    print "Creating embeddings"
    sys.argv = ["embeddParser.py", vocabFiles[0], vocab_size ,gloveFile, embedding_dimension]
    execfile("embeddParser.py")


print "Everything is prepared, running main script!"
sys.argv = ["translate.py", "--data_dir="+data_path, "--train_dir="+data_path, 
            "--vocab_size="+str(vocab_size), "--embedding_dimensions="+str(embedding_dimension)]
execfile("translate.py")
