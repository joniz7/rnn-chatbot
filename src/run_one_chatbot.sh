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
