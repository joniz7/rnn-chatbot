import sys
import os

print os.getcwd()
"""
runstats = open("./data/runstats", "r+")

stats = {}

for line in runstats:
  line = line.split()
  stats[line[0]] = int(line[1])
  print "line: %s    %s"%(line[0], line[1])

print stats

current = stats["current"]
stats["current"] = (current+1)%stats["last"]

runstats.seek(0)
runstats.truncate()

print stats

for key in stats:
  print key
  runstats.write("%s %d\n"%(key, stats[key]))

runstats.close()
"""
# lstm on, batch_size=64, sa hog size som mojligt

current = 1

sys.argv = ["translate.py", "--vocab_size=60000", "--size=2000", "--num_layers=2", 
"--batch_size=64", "--max_running_time=40", "--embedding_dimensions=300", "--max_patience=12000", "--dropout_keep_prob=0.5", "--use_lstm=True",
"--patience_sensitivity=0.5", "--num_samples=2048", "--learning_rate=0.1", "--steps_per_checkpoint=50", "--train_data_part="+str(current)]

execfile("./src/translate.py")

