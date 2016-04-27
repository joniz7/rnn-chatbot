cd checkpoints/$1

if [ls | wc -l > 3]
	then
		echo "Many checkpointfiles"
		ls -t | sed -e '1,3d' | xargs -d '\n' rm
fi

cd ../../src

shift
#$*