var send = $("#sendBut");
var input = $("#inpField");
var log = $("#chatlog");
var spin = $("#spinner");

send.css("background-color", "green");
var xhr = new XMLHttpRequest();

var ready = true;
spin.css("visibility", "hidden");

var serverIp = "128.199.46.170";
var serverPort = "3000";

$("form").on("submit", function(){
  return false;
});

send.click(function(){
  if(ready) {
    var msg = input.val();
    input.val("");
    addMessage(msg);

    $.ajax({
      url: "http://"+serverIp+":"+serverPort+"/bot/"+msg,
      jsonp: "cb",
      dataType: "jsonp",
   
      // Work with the response
      success: function( response ) {
        console.log(response);
        if(response.msg[0] == ">") response.msg = response.msg.substring(1);
        addMessage(response.msg);
        send.css("visibility", "visible");
        spin.css("visibility", "hidden");
        ready = true;
      }
    });

    send.css("visibility", "hidden");
    spin.css("visibility", "visible");
    ready = false;
  }
});

var i = 0;

function addMessage(msg) {
  if(i%2 == 0) {
    log.append('<div class="chatlogElem you"> you:&#09&#09 '+msg+"</div>");
  } else {
    log.append('<div class="chatlogElem bot"> bot:&#09&#09 '+msg+"</div>");
  }
  i++;
  log.animate({scrollTop:log[0].scrollHeight}, {duration:0});
}