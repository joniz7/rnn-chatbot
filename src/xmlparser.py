import xml.etree.ElementTree as ET
from datetime import datetime
import re
import os
import gzip
import sys
import time
import random

"""
to lowercase
remove brackets
(replace  with ')
replace dots with _DOTS
["<", ">","/","\\", "=", "--"]
  """

def main():
  userinput = ""

  if len(sys.argv) > 1:
    if len(sys.argv) >= 5:
      arguments = (sys.argv[1], sys.argv[2], int(sys.argv[3]), (sys.argv[5] if len(sys.argv)==6 else None), int(sys.argv[4]))
      print "Running with arguments \nfile: %s\ntarget: %s\nsplittime: %d\nbeginfile: %s\nskiplines: %d"%arguments
      if os.path.isdir(os.path.join(os.getcwd(), sys.argv[1])):
        parseDir(*arguments)
      else:
        with open(sys.argv[2], "w") as f: 
          f.write(parseFile(sys.argv[1], int(sys.argv[3]), skiplines = int(sys.argv[4])).encode("utf-8"))
    elif sys.argv[1] == "debug":
      with open("parsed.txt", "w") as f:
        f.truncate()
        f.write(parseFile("C:\\Exjobb\\parsing\\en\\1895\\309\\3195494_1of4.xml", 30, 10).encode("utf-8"))
    elif sys.argv[1] == "benchfile":
      testparams = ("C:\\Exjobb\\parsing\\en\\1930\\7291\\3369423_1of1.xml", 25)
      numruns = 1000
      open("time.txt", "w").truncate()

      totaltime = 0

      # Without skiplines
      start = datetime.now()
      for _ in range(numruns):
        parseFile(*testparams+(0,))
        
      totaltime = (datetime.now() - start).total_seconds()
      print "Without skiplines avg: %f"%(totaltime/numruns)

      # With skiplines
      start = datetime.now()
      for _ in range(numruns):
        parseFile(*testparams+(10,))
      
      totaltime = (datetime.now() - start).total_seconds()
      print "With skiplines avg: %f"%(totaltime/numruns)


    elif re.match(r"\b\d+\b", sys.argv[1]):
      parseDir("en", "parsed%d.txt"%(int(sys.argv[1])), int(sys.argv[1]), 10)

  while userinput.lower() != "e":
    print "What would you like to do?"
    print "1. Unzip files"
    print "2. Parse file"
    print "3. Parse files in folder"
    print "E. Exit"

    userinput = raw_input()

    if userinput == "1":
      unzipfiles(path=raw_input("Enter folder name:"))
    elif userinput == "2":
      filename = raw_input("Enter filename: ")
      targetfile = raw_input("Enter target filename:")
      splittime = int(raw_input("Enter time to split:"))
      skiplines = int(raw_input("How many lines do you want to skip?"))
      with open(targetfile, "w") as f:
        f.truncate()
        f.write(parseFile(filename, splittime, skiplines).encode("utf-8"))
    elif userinput == "3":
      dirname = raw_input("Enter dirname: ")
      targetfile = raw_input("Enter target filename:")
      splittime = int(raw_input("Enter time to split:"))
      beginfile = raw_input("Enter file to begin at (Nothing for beginning):")
      skiplines = int(raw_input("How many lines do you want to skip?"))
      with open(targetfile, "w") as f:
        f.truncate()
      parseDir(dirname, targetfile, splittime, skiplines=skiplines, beginfile=beginfile)

def splitfile(file, dists, filenames=None, splittoken="_NEWCONVO"):
  if not filenames:
    filenames = []
    for i, _ in enumerate(dists):
      filenames.append("%s%d"%(file, i))

  openfiles = []
  for fn in filenames:
    openfiles.append(open(fn, "w"))

  with open(file, "rb") as f:
    for line in f:
      print ""
      # las in allt tills _NEWCONVO
      # slumpa tal och valj ratt fil
      # skriv till filen

def generatestatistics(filename, targetfile):
  with open(filename, "rb") as scriptfile:
    numWords = 0
    numLines = 0
    numConvos = 0
    for line in scriptfile:
      if line.strip() != "_NEWCONVO":
        numLines = numLines + 1
        numWords = numWords + len(line.split())
      else:
        numConvos = numConvos + 1

  with open(targetfile, "w") as statfile:
    statfile.truncate()

    statfile.write("Total words: %d\n" +
                    "Total lines: %d\n"+
                    "Total convos: %d\n"+
                    "Avg words: %d\n"+
                    "Avg lines: %d"%(numWords, numLines, numConvos, numWords/numConvos, numLines/numConvos))


def unzipfiles(path=""):
  path = os.path.join(os.getcwd(), path)
  for (dirpath, _, files) in os.walk(path):
    print "unzipping "+dirpath
    for file in files:
      if file[-3:] == ".gz":
        with gzip.open(os.path.join(dirpath, file)) as f:
          writefile = open(os.path.join(dirpath,file[:-2]), "w")
          writefile.write(f.read())
        os.remove(os.path.join(dirpath, file))

def formatline(line):
  with open("removed.txt", "a") as f:
    f.truncate()
    # Remove dashes (-)
    if len(line) == 0:
      return line

    line = line.strip()
    line = line.replace("-", "")
    line = line.lower()
    # Regex that removes everythin within () or [] including the brackts
    regex = r"\(.*\)|\[.*\]"
    line = re.sub(regex, "", line)
    line = replaceDots(line)
    if(len(line)>1 and line[-1]==","):
      line = line[:-1]+"."

    # Remove urls
    if "www" in line or "http : //" in line or "https : //" in line:
      f.write(("removed "+line+"\n").encode("utf-8"))
      line = ""

    line = line.replace(".", ".\n")
    line = line.replace("!", "!\n")
    line = line.replace("?", "?\n")

    return line

# Replaces multiple dots or commas with _DOTS
def replaceDots(line):
  return re.sub("(\.|\,)(\.|\,)+", "_DOTS", line)

def timestringToInt(oldtimestamp):
  timestamp = fixtimestamp(oldtimestamp)
  try:
    date = datetime.strptime(timestamp, "%H:%M:%S,%f")
  except ValueError:
    print oldtimestamp
    print timestamp
    date = datetime.strptime(timestamp, "%H:%M:%S")

  return (date.hour*60 + date.minute)*60 + date.second

# TODO: When timestamp are incorrect, return and ignore
def fixtimestamp(timestamp):
  tempI = timestamp.index(':')
  hrs = int(timestamp[:tempI])
  timestamp = timestamp[tempI+1:]

  tempI = timestamp.index(':')
  mints = int(timestamp[:tempI])
  timestamp = timestamp[tempI+1:]

  try:
    tempI = timestamp.index(',')
  except ValueError:
    tempI = timestamp.find(':')
  
  if tempI == -1:
    secnds = int(timestamp)
  else:
    secnds = int(timestamp[0:tempI])
  timestamp = timestamp[tempI:]

  mints = mints + secnds / 60
  hrs = hrs + mints / 60
  secnds = secnds % 60
  mints = mints % 60
  hrs = hrs % 24
  
  timestamp = "0000" if not timestamp[1:] else timestamp

  return "%d:%d:%d,"%(hrs, mints,secnds)+timestamp[1:]

def parseDir(dirname, targetfile, splittime=30, beginfile=None, skiplines=0):
  if beginfile:
    print "Beginning at %s"%beginfile
    beginfile = os.path.join(os.getcwd(), beginfile)
  with open(targetfile, "a") as f:
    for (dirpath, _, files) in os.walk(os.path.join(os.getcwd(), dirname)):
      for file in files:
        try:
          filepath = os.path.join(dirpath, file)
          if not beginfile or beginfile == filepath:
            beginfile = None
            print "Parsing %s"%filepath
            f.write(parseFile(filepath, splittime=splittime, skiplines=skiplines).encode("utf-8"))
            f.flush()
        except KeyboardInterrupt:
          print "Exited during "+filepath

          with open("restart.py", "w") as restartfile:
            restartfile.truncate()
            restartfile.write("import sys\n")
            # Since write writes all \\ as \
            filepath = filepath.replace("\\", "\\\\")
            restartfile.write('sys.argv = ["parser.py", "%s", "%s", "%d", "%d", "%s"]\n'%(dirname, targetfile, splittime, skiplines,filepath))
            restartfile.write('execfile("parser.py")')
            print "Restart file generated. Run restart.py to start where we left off."
          f.flush()
          sys.exit(0)

def parseFile(filename, splittime=30, skiplines=0):
  tree = ET.parse(filename)
  root = tree.getroot()
  timestamp = 0
  timedifference = 0
  writesentence = ""

  """
  One line can stretch between multiple ids
  One id can hold multiple lines
  Make sure it works
  """

  movietext = ""

  i = skiplines

  for sentence in root:
    if i <= 0:
      for word in sentence:
        if word.tag == "time":
          if word.attrib["id"][-1] == "E" and re.match(r"\b\d+:\d+:\d+(,\d+)?$", word.attrib["value"]):
            timestamp = timestringToInt(word.attrib["value"])
          elif word.attrib["id"][-1] == "S" and re.match(r"\b\d+:\d+:\d+(,\d+)?$", word.attrib["value"]):
            timedifference = timestringToInt(word.attrib["value"]) - timestamp
            if timedifference < 0:
              print "timedifference %d movie %s"%(timedifference, filename)
            if timedifference > 30:
              movietext = movietext + "_NEWCONVO\n"
            formattedLine = formatline(writesentence)
            if formattedLine:
              movietext = movietext + formattedLine.strip()+"\n"
            writesentence = ""
        elif word.tag == "w":
            if word.text:
              writesentence = writesentence + " " + word.text
    else:
      i = i-1

  for _ in range(skiplines):
    movietext = movietext[:movietext.rfind("\n")]


    return movietext
  
def parsetimes(filename, targetfile):
  tree = ET.parse(filename)
  root = tree.getroot()

  with open(targetfile, "a") as f:
    timestamp = 0
    timedifference = 0

    for sentence in root:
      for word in sentence:
        if word.tag == "time":
          if word.attrib["id"][-1] == "E":
            timestamp = timestringToInt(word.attrib["value"])
          elif word.attrib["id"][-1] == "S":
            timedifference = timestringToInt(word.attrib["value"]) - timestamp
            f.write("%d\n"%timedifference)

def searchword(searchw, dir="en", beginfile=None):
  searchw = searchw.lower()
  for (dirpath, _, files) in os.walk(os.path.join(os.getcwd(), dir)):
    print "Searching in %s"%dirpath
    for file in files:
      filename = os.path.join(dirpath, file)
      if not beginfile or beginfile == filename:
        beginfile = None
        tree = ET.parse(filename)
        root = tree.getroot()
        for sentence in root:
          searchs = ""
          for word in sentence:
            if word.tag == "w":
              if word.text:
                searchs = searchs + " %s"%word.text.lower().strip()

          if searchs.strip() == searchw.strip():
            print "Word found in file: %s"%filename

if __name__ == "__main__":
  main()
  #searchword("you 're just lucky all six of them are still alive .", beginfile="C:\\Exjobb\\parsing\\en\\1990\\25\\33296_1of2.xml")