import os

import pyAIML as aiml
import sys

k = aiml.Kernel()

os.chdir("aiml")

for _, _, files in os.walk(os.getcwd()):
  for file in files:
    k.learn(file)

while True: print k.respond(sys.stdin.readline())