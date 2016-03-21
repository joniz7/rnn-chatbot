var PythonShell = require('python-shell');
var readline    = require('readline');
var express     = require('express');
var prompt      = require('prompt');
var process     = require('process');
var fs          = require('fs');

var chatbots = {};

process.chdir("src");
chatbots["cleverbot"] = new PythonShell("clever.py", {pythonOptions:["-u"]});
chatbots["alice"]     = new PythonShell("alice.py", {pythonOptions: ["-u"]});
chatbots["arenen"]    = new PythonShell("translate.py", {pythonOptions: ["-u"], args:["--size=2550","--num_samples=2048","--batch_size=1","--decode_randomness=0.3", "--vocab_size=100000", "--num_layers=2", "--decode=True"]});
var app = express();

var curRes = undefined;
var curBot = undefined;
var manual = false;
app.set("jsonp callback name", "cb");

var time = new Date();

var filepath = 'data/chatlogs/log'+time.getTime()+'.txt';
fs.open(filepath, "w");

function writeFile(msg) {
  fs.appendFile(filepath, msg+"\n");
}

app.get("/bot/:message", function(req, res){
  console.log("client: "+req.params.message);
  writeFile("client: "+req.params.message);
  curRes = res;
  if(curBot) {
    curBot.send(req.params.message);
  } else if(!manual) {
    res.status(200).jsonp({msg: "No chatbot available right now :("})
  }
});

app.listen(3000, function(){
  console.log("it is up, go to localhost:3000");
});

for(var name in chatbots){
  new function(n){
    chatbots[n].on('message', function(message){
    if(curRes) {
      returnMessage(message);
    } else {
      console.log("from "+n+": "+message);
    }
    });
    chatbots[n].on('error', function(error){
      console.log(error);
    });
  }(name);
}

function returnMessage(message) {
  if(curRes) {
    curRes.status(200).jsonp({msg: message});
    writeFile("server: "+message);
  }
}

prompt.start();
prompt.get('bot', getInput);
function getInput(err, result) {
  if(manual) {
    if(result == "exit") {
      manual = false
    } else if(curRes) {
      console.log("sent message");
      returnMessage(result.bot);
    }
  } else {
    console.log(result.bot + " picked");
    if(chatbots[result.bot]) {
      curBot = chatbots[result.bot];
      writeFile("SWITCHED TO "+result.bot);
    } else if(result.bot == "manual") {
      manual = true;
      curBot = undefined;
      writeFile("SWITCHED TO manual");
    } else {
      console.log("no such bot");
    }
  }
  
  prompt.get('bot', getInput);
}