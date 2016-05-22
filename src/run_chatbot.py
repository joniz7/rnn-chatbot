import sys

runstats = open("runstats", "w")

stats = {}

for line in runstats:
  line = line.split()
  stats[line[0]] = int(line[1])

current = line[current]
line[current] = (current+1)/line[last]

runstats.truncate()

for key in stats:
  runstats.writeline("%s %d"%(key, stats[key]))

sys.argv = ["translate.py", "--vocab_size=30000", "--size=1024", "--num_layers=3", 
"--batch_size=64", "--max_running_time=40", "--embedding_dimensions=300", "--max_patience=12000",
"--patience_sensitivity=0.5", "--num_samples=2048", "--train_data_part="+current]

execfile("./src/translate.py")

