from __future__ import print_function
import cleverbot
import sys


cb = cleverbot.Cleverbot()

sys.stdout.flush()
while True:
  inp = sys.stdin.readline()
  ans = cb.ask(inp)
  print(ans)
  #print "received"
  sys.stdout.flush()

