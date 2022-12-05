var StartTime = Date.now();
var today = new Date();
var dd = today.getDate() - 1;
if(dd<10){dd='0'+dd.toString();} else{dd=dd.toString();}
var mm = today.getMonth() + 1; //January is 0!
if(mm<10){mm='0'+mm.toString();} else{mm=mm.toString();}
var yyyy = today.getFullYear();
yyyy=yyyy.toString()

function initDemoMap(){
//BASE TILE LAYER 1
  var Esri_WorldImagery = L.tileLayer('http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, ' +
    'AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
  });
//BASE TILE LAYER 2
  var Esri_DarkGreyCanvas = L.tileLayer(
    "http://{s}.sm.mapstack.stamen.com/" +
    "(toner-lite,$fff[difference],$fff[@23],$fff[hsl-saturation@20])/" +
    "{z}/{x}/{y}.png",
    {
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, ' +
      'NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community'
    }
  );
//BASE TILE LAYER 3
  var Stamen_Toner = L.tileLayer('http://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.{ext}', {
    attribution: 'Map tiles by Stamen Design, CC BY 3.0 &mdash; Map data &copy; OpenStreetMap',
    subdomains: 'abcd',
    minZoom: 0,
    maxZoom: 20,
    ext: 'png'
  });
//BASE TILE LAYER 4
  var Esri_ShadedRelief = L.tileLayer('http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri'
  });
//
  var Esri_Oceans = L.tileLayer('http://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Sources: Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE, Geonames.org, and other contributors'
  });
  var Esri_Topo = L.tileLayer('http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Sources: Esri, HERE, DeLorme, Intermap, increment P Corp., GEBCO, USGS, FAO, NPS, NRCAN, GeoBase, IGN,' +
	' Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), swisstopo, MapmyIndia, &copy;  OpenStreetMap' +
	' contributors, and the GIS User Community'
  });

//BASE TILE GROUP LAYER
  var baseLayers = {
    "Relief": Esri_ShadedRelief,
    "Satellite": Esri_WorldImagery,
    "Oceans ": Esri_Oceans,
    "Grey ": Esri_DarkGreyCanvas,
    "Topo ": Esri_Topo,
  };
//MAP STRUCTURE
  var map = L.map('map', {
    layers: [ Esri_ShadedRelief ],
    minZoom : 1,
    worldCopyJump: true,
    inertia: false
  });

//MENU CREATION
  var layerControl = L.control.layers(baseLayers);
  layerControl.addTo(map);
  map.setView([20, -45], 2);
//   map.fitWorld(maxZoom=2);

//MOUSE POSITION BOTTOM LEFT
  L.control.mousePosition().addTo(map);

//INIT RETURN FUNCTION
  return {
    map: map,
    layerControl: layerControl
  };
}

// MAP CREATION
var mapStuff = initDemoMap();
var map = mapStuff.map;
var layerControl = mapStuff.layerControl;


//TRAJ LAYER, EMPTY AT START
var majaxLayer=L.layerGroup();
var majaxLayerLine=L.layerGroup();
map.addLayer(majaxLayer);
//CADDY LAYER, EMPTY AT START
var caddyLayer=L.layerGroup();
map.addLayer(caddyLayer);

//DATA LAYERS

//SST VIA CMEMS WMS
var wmsLayer0 = L.tileLayer.wms('http://nrt.cmems-du.eu/thredds/wms/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2?', {
   layers: 'analysed_sst',
   opacity: 0.45,
   colorscalerange: '271.0,303.0',
   abovemaxcolor: "extend",
   belowmincolor: "extend",
   numcolorbands: 30,
   time: yyyy+'-'+mm+'-'+dd+'T12:00:00.000Z',
   styles: 'boxfill/rainbow'
});
// htmlsst='<font color="magenta">SST '+yyyy+'-'+mm+'-'+dd+'</font> <a target="_blank" href="http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=SST_GLO_SST_L4_NRT_OBSERVATIONS_010_014"><img src="static/dist/info.png" height="15" width="15"></a>';
htmlsst='<font color="magenta">SST '+yyyy+'-'+mm+'-'+dd+'</font>';
Spansst="<span id='ssttag'>"+htmlsst+"</span>"
layerControl.addOverlay(wmsLayer0, Spansst, "SST");

//SLA VIA CMEMS WMS
var wmsLayer0 = L.tileLayer.wms('https://nrt.cmems-du.eu/thredds/wms/global-analysis-forecast-phy-001-024?', {
   layers: 'zos',
   opacity: 0.45,
   colorscalerange: '-1.0,1.0',
   abovemaxcolor: "extend",
   belowmincolor: "extend",
   numcolorbands: 30,
   time: yyyy+'-'+mm+'-'+dd+'T12:00:00.000Z',
   styles: 'boxfill/redblue'
});
// htmlsla='<font color="magenta">SLA '+yyyy+'-'+mm+'-'+dd+'</font> <a target="_blank" href="https://data.marine.copernicus.eu/product/GLOBAL_ANALYSIS_FORECAST_PHY_001_024"><img src="static/dist/info.png" height="15" width="15"></a>';
htmlsla='<font color="magenta">SLA '+yyyy+'-'+mm+'-'+dd+'</font>';
Spansla="<span id='slatag'>"+htmlsla+"</span>"
layerControl.addOverlay(wmsLayer0, Spansla, "SLA");

//MDT VIA CMEMS WMS
var wmsLayer0 = L.tileLayer.wms('https://my.cmems-du.eu/thredds/wms/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y?', {
   layers: 'mdt',
   opacity: 0.45,
   colorscalerange: '-1.0,1.0',
   abovemaxcolor: "extend",
   belowmincolor: "extend",
   numcolorbands: 30,
   time: '2003-01-01T00:00:00.000Z',
   styles: 'boxfill/redblue'
});
// htmlmdt='<font color="magenta">MDT CNES-CLS18_CMEMS2020</font> <a target="_blank" href="https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_MDT_008_063"><img src="static/dist/info.png" height="15" width="15"></a>';
htmlmdt='<font color="magenta">MDT CNES-CLS18_CMEMS2020</font>';
Spanmdt="<span id='slatag'>"+htmlmdt+"</span>"
layerControl.addOverlay(wmsLayer0, Spanmdt, "MDT");

function getcolorfor(value){
    // Check: http://google.github.io/palette.js to pick a colormap
    // var pal_for_errors = palette(['cb-Blues'], 5, 1);
    var n = 5
    var pal_for_errors = palette(['tol-sq'], n, 1);
//     ic = n-4
//     if (value < 4) {ic=n-3};
//     if (value < 3) {ic=n-2};
//     if (value < 2) {ic=n-1};
//     if (value < 1) {ic=n-0};
    ic = n-1
    if (value < 4) {ic=n-2};
    if (value < 3) {ic=n-3};
    if (value < 2) {ic=n-4};
    if (value < 1) {ic=0};
    return pal_for_errors[ic]
}

function getpopupcontent(data){
    var Bearing = Math.round(100*data['prediction_location_error']['bearing']['value'])/100;
    var Distance = Math.round(100*data['prediction_location_error']['distance']['value'])/100;
    var Time = Math.round(100*data['prediction_location_error']['time']['value'])/100;
    var Transit = Math.round(100*data['prediction_metrics']['transit']['value'])/100;
    var WMO = data['previous_profile']['wmo'];
    var nfloats = data['meta']['Nfloats'];
    var velocity = data['meta']['Velocity field'];

    html = '<div class="card">'

    html += '<div class="card-header">'
    html += '   <ul class="nav nav-tabs card-header-tabs">'
    html += '       <li class="nav-item">'
    html += '           <a class="nav-link active" aria-current="true" href="#">Error</a>'
    html += '       </li>'
    html += '       <li class="nav-item">'
    html += '           <a class="nav-link" href="'+url_wmomap.replace('/map_error', '/map_error/'+WMO)+'">Show only this float</a>'
    html += '       </li>'
    html += '       <li class="nav-item">'
    html += '           <a class="nav-link" href="'+data['meta']['api']['cycle_page']+'">Cycle page</a>'
    html += '       </li>'
    html += '       <li class="nav-item">'
    html += '           <a class="nav-link" href="'+data['meta']['api']['float_page']+'">Swipe this</a>'
    html += '       </li>'
    html += '   </ul>'
    html += '</div>'

//     html += '<img src="'+data['meta']['figures']['predictions_recap']+'" class="card-img-top" alt="">'
    html += '<img src="'+data['meta']['figures']['predictions']+'" class="card-img-top" alt="">'

    html += '<div class="card-body">'
    html += '   <p class="card-text">Bearing: <span class="text-muted fw-light">'+Bearing+' '+data['prediction_location_error']['bearing']['unit']+'</span>'
    html += '   <br>Distance: <span class="text-muted fw-light">'+Distance+' '+data['prediction_location_error']['distance']['unit']+'</span>'
    html += '   <br>Time: <span class="text-muted fw-light">'+Time+' '+data['prediction_location_error']['time']['unit']+'</span>'
    html += '   <br>Transit: <span class="text-muted fw-light">'+Transit+' '+data['prediction_metrics']['transit']['unit']+'</span></p>'
    html += '</div>'

    html += '<div class="card-footer">'
    html += '   <small class="text-muted">Computed in '+ data['meta']['Computation']['Wall-time']+' on '+data['meta']['Computation']['system']['hostname']+'</small>'
    html += '</div>'

    html += '</div>'

    return html
}

function circle_props(feature){
    var dist = 1000*feature['properties']['prediction_location_error']['distance']['value'];
    var transit = Math.round(100 * feature['properties']['prediction_metrics']['transit']['value'])/100;
//     console.log(transit);
    if (transit == 0){
        var color = '#000';
    } else {
        var color = '#'+getcolorfor(transit);
    }
    return {radius: dist, color: color, properties: feature.properties}
}

// Add default markers where we have predictions:
// var floatmarkers = L.layerGroup();
var floatcircles = L.layerGroup();
var LON = new Array();
var LAT = new Array();
var coords = new Array();
var Npoints = 0
$.getJSON(url_data, function(data) {
    for (feature in data['features']) {
//         console.log(data['features'][feature])
//         console.log(data['features'][feature]['properties']['prediction_location_error']['bearing']['value'])
//         console.log(data['features'][feature]['properties']['prediction_location_error']['distance']['value']);
//         console.log(data['features'][feature]['properties']['prediction_metrics']['transit']['value']);

        Npoints += 1
        var this_feature = data['features'][feature]
//         marker = L.geoJSON(this_feature, {
// //             style: function (feature) {
// //                 return {color: feature.properties.color};
// //             },
//         }).bindPopup(function (layer) {
//             return "Transit:" + layer.feature.properties.prediction_metrics.transit.value;
//         })
//         marker = L.circle([this_feature.geometry.coordinates[1], this_feature.geometry.coordinates[0]], {radius:1, color: "#000"});
//         marker.on('click', L.bind(SubMarkerClick, null, this_feature));
//         marker.addTo(floatmarkers);
//         console.log(this_feature.geometry.coordinates[0])
        LON.push(this_feature.geometry.coordinates[0])
        LAT.push(this_feature.geometry.coordinates[1])
        coords.push([this_feature.geometry.coordinates[1], this_feature.geometry.coordinates[0]])

        circle = L.circle([this_feature.geometry.coordinates[1], this_feature.geometry.coordinates[0]], circle_props(this_feature));
        circle.bindPopup(function (layer) {
            var props = layer.options.properties;
            var html = getpopupcontent(props);
            return html
        }, {minWidth : 560});
        circle.addTo(floatcircles);
    }
}).done(function() {
    var bounds = new L.LatLngBounds(coords);
    if ( (bounds.getEast()-bounds.getWest() < 0.5) || (bounds.getNorth()-bounds.getSouth() < 0.5) ) {
        var LONmean = LON.reduce((a, b) => a+b, 0)/LON.length;
        var LATmean = LAT.reduce((a, b) => a+b, 0)/LAT.length;
        map.setView([LATmean, LONmean], 5);
    } else {
        map.fitBounds(bounds);
    };
    $("#Npoints").text(Npoints + ' predictions');
}).fail(function( jqxhr, textStatus, error ){
    var err = textStatus + ", " + error;
    console.log('error when loading data from '+url_data);
    console.log( "Request Failed: " + err );
});
// floatmarkers.addTo(map);
floatcircles.addTo(map);
// L.circle([48.41, -4.47], {radius: 10*1000}).addTo(map); # Brest circle for reference

var legend = L.control({position: 'bottomright'});
legend.onAdd = function (map) {
    var div = L.DomUtil.create('div', 'info legend'),
        grades = [0, 1, 2, 3, 4],
        labels = [];
    // loop through our density intervals and generate a label with a colored square for each interval
    for (var i = 0; i < grades.length; i++) {
        div.innerHTML +=
            '<i style="background: #' + getcolorfor(grades[i]) + '"></i> ' +
            grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + ' hours <br>' : '+');
    }
    return div;
};
legend.addTo(map);

if (trajdata != ''){
    L.geoJSON(trajdata).addTo(map);
}