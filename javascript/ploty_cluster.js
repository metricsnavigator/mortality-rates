/*
Be sure to reference the plotly javascript source in your HTML header. 
Feel free to borrow the base file from here: <script src="https://cdn.plot.ly/plotly-2.16.1.min.js"></script>
*/

<div id="scatter_1"></div>
<script>
CLUSTERS = document.getElementById('scatter_1');
var trace1 = {
  x: [47,53,52,35,31,42,46,50,60,45,54,44],
  y: [982,1030,1017,860,891,971,952,1006,861,961,1113,1025],
  mode: 'markers',
  type: 'scatter',
  name: 'Cluster 1',
  text: [],
  marker: {size:9}
};
var trace2 = {
  x: [36,35,44,43,43,45,36,36,33,40,35,37,36,31,30,31,43,39,35,43,30,30,46,42,42,36,42,41,32,34,45,38,31,40,45,45,42,38],
  y: [922,998,962,1071,935,900,1002,912,1025,970,986,959,936,959,941,871,887,969,920,844,989,929,923,995,1015,991,938,946,874,954,904,951,972,912,1004,896,912,954],
  mode: 'markers',
  type: 'scatter',
  name: 'Cluster 2',
  text: [],
  marker: {size:9}
};
var trace3 = {
  x: [15,11,25,37,10,18,13,35,41,28],
  y: [872,862,858,894,840,912,791,899,968,824],
  mode: 'markers',
  type: 'scatter',
  name: 'Cluster 3',
  text: [],
  marker: {size:9}
};
var data = [trace1,trace2,trace3];
var layout = {
  xaxis: {
    range: [5,70]
  },
  yaxis: {
    range: [800,1150]
  },
  showlegend: true,
  legend: {"orientation": "h"},
  title:'Precipitation & Mortality'
};
var config = {responsive: true}
Plotly.newPlot('scatter_1',data,layout,config);
</script>

