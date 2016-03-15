var PythonShell = require('python-shell');
var readline    = require('readline');
var express     = require('express');
var process     = require('process');
var fs          = require('fs');
var _           = require('lodash');
var shortid     = require('shortid');
var bodyParser  = require('body-parser');

var chatbot = new PythonShell("src/translate.py", {pythonOptions: ["-u"], args:["--embedding_dimensions=300", "--num_layers=5", "--decode=True", "--decode_randomness=0.2"]});
var app = express();

app.use(bodyParser.urlencoded({extended: false})); 
app.use(bodyParser.json());

// Add headers
app.use(function (req, res, next) {

    // Website you wish to allow to connect
    res.setHeader('Access-Control-Allow-Origin', 'http://128.199.46.170:59254');

    // Request methods you wish to allow
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');

    // Request headers you wish to allow
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');

    // Set to true if you need the website to include cookies in the requests sent
    // to the API (e.g. in case you use sessions)
    res.setHeader('Access-Control-Allow-Credentials', true);

    // Pass to next layer of middleware
    next();
});

console.log("no port :(");

var resQ = {};

app.set("jsonp callback name", "cb");

var time = new Date();
var filepath = 'data/chatlogs/log'+time.getTime()+'.txt';

function writeFile(msg) {
  fs.appendFile(filepath, msg+"\n");
}

app.post("/", function(req, res){
  var message = req.body.msg;
  console.log(req.body);
  console.log("client "+req.ip+": "+message);
  
  // writeFile("client: "+message);
  var uniqueId = shortid.generate();
  chatbot.send(uniqueId + " " + message);
  console.log(uniqueId + " " + message);
  resQ[uniqueId] = res;
});

app.listen(3000, function(){
  console.log("it is up, go to localhost:3000");
});

chatbot.on('message', function(message){
  console.log("chatbot: ");
  var splitMsg = message.split(" ");
  var key = splitMsg[0];
  message = _.reduce(splitMsg.slice(1), function(a, b){return a+" "+b}, "");
  console.log("key "+ key);
  if(resQ[key]) {
    console.log("message sent", message);
    resQ[key].json({msg: message});
    delete resQ[key];
  } else {
    console.log(message);
  }
});
chatbot.on('error', function(error){
  console.log(error);
});