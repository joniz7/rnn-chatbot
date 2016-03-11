var PythonShell = require('python-shell');
var readline    = require('readline');
var express     = require('express');
var process     = require('process');
var fs          = require('fs');
var _           = require('lodash');
var shortid     = require('shortid');

var chatbot = new PythonShell("src/translate.py", {pythonOptions: ["-u"], args:["--embedding_dimensions=300", "--size=5", "--num_layers=1", "--decode=True"]});
var app = express();

var resQ = {};

app.set("jsonp callback name", "cb");

var time = new Date();
var filepath = 'data/chatlogs/log'+time.getTime()+'.txt';

function writeFile(msg) {
  fs.appendFile(filepath, msg+"\n");
}

app.get("/:message", function(req, res){
  console.log("client "+req.ip+": "+req.params.message);
  
  // writeFile("client: "+req.params.message);
  var uniqueId = shortid.generate();
  chatbot.send(uniqueId + " " + req.params.message);
  console.log(uniqueId + " " + req.params.message);
  resQ[uniqueId] = res;
});

app.listen(3000, function(){
  console.log("it is up, go to localhost:3000");
});

chatbot.on('message', function(message){
  console.log(message);
  var splitMsg = message.split(" ");
  var key = splitMsg[0];
  message = _.reduce(splitMsg.slice(1), function(a, b){return a+" "+b}, "");
  console.log(key);
  if(resQ[key]) {
    resQ[key].jsonp({msg: message})
    delete resQ[key];
  } else {
    console.log(message);
  }
});
chatbot.on('error', function(error){
  console.log(error);
});