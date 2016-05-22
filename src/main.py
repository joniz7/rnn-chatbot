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

"""
Following flags can be set when run:
--vocab_size
--embedding_dimension
--embeddings_filename
--data_path
"""

# Parameters
vocab_size = 90000
embedding_dimension = 300
data_path = "../data"
batch_size = 64
size = 1024
num_layers = 3
decode = False
split_sentence = False
max_running_time=40 
max_patience=12000
patience_sensitivity=0.5

for arg in sys.argv[1:]:
  arg = arg.split("=")
  if(arg[0][:2] == "--"):
    if(arg[0][2:] == "vocab_size"):
      print "vocab_size "+arg[1]
      vocab_size = int(arg[1])
    elif(arg[0][2:] == "embedding_dimensions"):
      print "embedding_dimensions "+arg[1]
      embedding_dimension = int(arg[1])
    elif(arg[0][2:] == "embeddings_filename"):
      print "embeddings_filename "+arg[1]
      embeddings_filename = str(arg[1])
    elif(arg[0][2:] == "data_path"):
      print "data_path "+arg[1]
      data_path = int(arg[1])
    elif(arg[0][2:] == "batch_size"):
      print "batch_size "+arg[1]
      batch_size = int(arg[1])
    elif(arg[0][2:] == "size"):
      print "size "+arg[1]
      size = int(arg[1])
    elif(arg[0][2:] == "num_layers"):
      print "num_layers "+arg[1]
      num_layers = int(arg[1])
    elif(arg[0][2:] == "decode"):
      print "decode "+arg[1]
      decode = bool(arg[1])
    elif(arg[0][2:] == "split_sentence"):
      print "split_Sentence "+arg[1]
      split_sentence = bool(arg[1])
    elif(arg[0][2:] == "max_running_time"):
      print "max_running_time "+arg[1]
      max_running_time=int(arg[1])
    elif(arg[0][2:] == "max_patience"):
      print "max_patience "+arg[1]
      max_patience = int(arg[1])
    elif(arg[0][2:] == "patience_sensitivity"):
      print "patience_sensitivity "+arg[1]
      patience_sensitivity = float(arg[1])
    else:
      print "Bad format on flag %s"%arg[0]
      sys.exit()
  else:
    print "Bad format on flag %s"%arg[0]
    sys.exit()

embeddings_filename = "embeddings"+str(vocab_size)+".txt"

# percentage of partitions between training, validation and test data respectively
train_data_percentage = 79
valid_data_percentage = 7
test_data_percentage = 14

print "Running main with parameters:"
print "vocab_size: %d\nembedding_dimension: %d\nembeddings_filename: %s\ndata_path: %s\ndecode: %s"%(vocab_size, embedding_dimension, embeddings_filename, data_path, str(decode))

#trainingFiles = [data_path+"/train-data.utte", data_path+"/train-data.resp", data_path+"/valid-data.utte", data_path+"/valid-data.resp"]
trainingFiles = [data_path+"/train-data.data", data_path+"/valid-data.data"]
print "========== Checking if training and validation data exists =========="
if(reduce(operator.and_, map(os.path.isfile, trainingFiles))):
  print "Exists, moving on"
else:
  sys.argv = ["browse.py", train_data_percentage, valid_data_percentage, test_data_percentage, False, split_sentence]
  execfile("browse.py")

vocabFiles = [data_path+"/vocab"+str(vocab_size), data_path+"/vocab"+str(vocab_size)+".utte"]
print "================== Checking if vocabulary exists ===================="

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
    sys.argv = ["embeddParser.py", vocabFiles[0], vocab_size ,gloveFile, embedding_dimension, data_path+"/"+embeddings_filename]
    execfile("embeddParser.py")


print "Everything is prepared, running main script!"
sys.argv = ["translate.py", "--data_dir="+data_path, "--train_dir="+data_path, 
            "--vocab_size="+str(vocab_size), "--embedding_dimensions="+str(embedding_dimension), 
            "--size="+str(size), "--num_layers="+str(num_layers), "--batch_size="+str(batch_size), "--decode="+str(decode)]
execfile("translate.py")
