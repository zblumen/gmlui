var express = require("express");
var elements = require('./pred_graph_3e45d4c24d91c3cce969890ceb79ffe5.json');
var app = express();


app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});
app.get("/cy_elements", function(req, res, next) {
  res.send(elements);
});

app.listen(5000, () => console.log('Example app listening on port 3000!'))