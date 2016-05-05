echo "run for $1"

python run_one_chatbot.py $1

shift
echo $*
$*
