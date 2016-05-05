"""
echo "run for $1"
cd checkpoints/$1

if [[ $(ls | wc -l) -gt 3 ]]
  then
    echo "Many checkpointfiles"
    ls -t | sed -e '1,9d' | xargs -d '\n' rm
fi

cd ../..

shift
echo $*
$*
"""

import os
import sys

os.chdir("checkpoints/"+sys.argv[1])

ckpt_begin = "translate.ckpt-"
meta_end = ".meta"

largest = 0

for _, _, files in os.walk(os.getcwd()):
  for file in files:
    if ckpt_begin in file and not meta_end in file:
      ckpt_num = int(file[len(ckpt_begin):])
      if ckpt_num > largest:
        largest = ckpt_num

  for file in files:
    if str(largest) not in file:
      os.remove(file)

with open("checkpoint", "w") as ckpt_f:
  ckpt_f.truncate()
  ckpt_f.write('model_checkpoint_path: "translate.ckpt-%d"\nall_model_checkpoint_paths: "translate.ckpt-%d"'%(largest, largest))