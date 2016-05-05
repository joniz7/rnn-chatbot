echo "run for $1"

python ./src/remove_checkpointfiles.py $1

shift
echo $*
$*
