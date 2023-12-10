
var currentSampleIdx = 0;
var currentTrainLabel = -1;
var configListGlobal = [];
var traceMap = new Map();

function evaluateSampleAt(idx) {
    $.ajax({
        type: "POST",
        url: "/evaluate_sample/" + idx,
        success: function(response) {
            var figureData = JSON.parse(response.figure);
            var audioPath = response.audio;

            // Use Plotly to update the figure
            Plotly.react('evalGraph', figureData.data, figureData.layout);

            // Update the source of the audio player
            var audioPlayer = document.getElementById('audioPlayer');
            audioPlayer.src = audioPath;
        }
    });
}

function evaluateSame() {
    evaluateSampleAt(currentSampleIdx);
}

function evaluateRandom() {
    currentSampleIdx = Math.floor(Math.random() * 100000);
    evaluateSampleAt(currentSampleIdx);
}

function evaluateNext() {
    currentSampleIdx++;
    evaluateSampleAt(currentSampleIdx);
}

function evaluatePrev() {
    currentSampleIdx = Math.max(0, currentSampleIdx - 1);
    evaluateSampleAt(currentSampleIdx);
}

// listeners
var checkpointSavedListener = function(event) {

}

var updateVizListener = function(event) {
    evaluateSame();
}

var trainingListener = function(event) {
    var updateData = JSON.parse(event.data);
    var width = 100 * (updateData.epoch + 1) / updateData.numEpochs;

    document.getElementById("trainingProgress").style.width = Number(width).toFixed(3) + '%';
    document.getElementById("trainingProgress").innerHTML = Number(width).toFixed(1) + '%';
    document.getElementById("trainingProgressInfo").innerHTML = `Epoch: ${updateData.epoch + 1}, Step: ${updateData.step}, Loss: ${Number(updateData.loss).toFixed(6)}`;
};

var loadingListener = function(event) {
    var updateData = JSON.parse(event.data);
    var width = 100 * (updateData.index + 1) / updateData.size;

    document.getElementById("loadProgress").style.width = Number(width).toFixed(3) + '%';
    document.getElementById("loadProgress").innerHTML = Number(width).toFixed(1) + '%';
    document.getElementById("loadProgressInfo").innerHTML = `Loaded ${updateData.index + 1} of ${updateData.size} samples`;
};

var loadEndListener = function(event) {
    var updateData = JSON.parse(event.data);
    var elapsed = updateData.elapsed;
    document.getElementById("loadProgress").innerHTML = "Finished";
};

function modelConfigSelected(selectObject) {
    var folder = selectObject.value;
    refreshConfigSelection(folder);
}

// plots
function updateTrainTrace(event) {
    var traceIdx = traceMap.get(currentTrainLabel);
    var updateData = JSON.parse(event.data);
    Plotly.extendTraces('lossGraph', {
        y: [[updateData.loss]],
        x: [[updateData.step]]
    }, [traceIdx]);
}

function updateEvalTrace(event) {
    var traceIdx = traceMap.get(currentTrainLabel);
    var updateData = JSON.parse(event.data);
    Plotly.extendTraces('lossGraph', {
        y: [[updateData.loss]],
        x: [[updateData.step]]
    }, [traceIdx + 1]);
}

function createLossGraph() {
    var layout = {
        title: 'Training and Evaluation Loss',
        xaxis: { title: 'Step' },
        yaxis: { title: 'Loss', type: 'log', domain: [0.0001, 1] },
        plot_bgcolor: 'rgba(0.3, 0.3, 0.3, 0.0)',
        paper_bgcolor: 'rgba(0.15, 0.15, 0.17, 1.0)',
        showlegend: false,
        template: 'plotly_dark',
        margin: {l:50, r:30, t:50, b:30},
        font: { color: '#888'}
    };

    Plotly.newPlot('lossGraph', [], layout).then(r => {
        console.log(r.data);
    });
}

function printMap() {
  for (const [key, value] of traceMap) {
    console.log(`${key}: ${value}`);
  }
}

function bumpUpValuesBy2(map) {
  const keysToUpdate = [];

  for (const [key, value] of map) {
    keysToUpdate.push(key);
  }

  for (const key of keysToUpdate) {
    const value = map.get(key);
    map.set(key, value + 2);
  }
}


function bumpDownTraceIndices(map, threshold) {
  const keysToUpdate = [];

  for (const [key, value] of map) {
    if (value >= threshold) {
      keysToUpdate.push(key);
    }
  }

  for (const key of keysToUpdate) {
    const value = map.get(key);
    map.set(key, value - 2);
  }
}

function updateTraces(hash, runIdx, isAdded) {
    var label = `${hash} ${runIdx}`;

    if(isAdded) {
        var row = configListGlobal.find(e => e.hash === hash && e.runIdx == runIdx);
        if(!! row?.trainingRun?.train) {
            var tr = row.trainingRun;
            var newTraces = [];
            var trainTrace = {
                x: tr.train.map(pair => pair[0]),
                y: tr.train.map(pair => pair[1]),
                mode: 'lines',
                name: `${label} - Train`,
//                line: { color: 'rgba(0.2, 0.2, 0.2, 1.0)' }
            };

            var evalTrace = {
                x: tr.eval.map(pair => pair[0]),
                y: tr.eval.map(pair => pair[1]),
                mode: 'lines',
                name: `${label} - Eval`,
//                line: { color: 'grey' }
            };

            newTraces.push(trainTrace, evalTrace);

            console.log(`Adding ${newTraces.length} new traces`);
            Plotly.addTraces('lossGraph', newTraces, [0, 1]).then(r => {
                bumpUpValuesBy2(traceMap);
                traceMap.set(label, 0);

                console.log(`Trace map after:`);
                printMap();

                console.log(r.data)
            });
        }
    } else if(traceMap.has(label)) {
        console.log("Label, idx: ", idx, label);

        var idx = traceMap.get(label);

        Plotly.deleteTraces('lossGraph', [idx, idx+1]).then(r => console.log(r.data));

        console.log(`Deleting traces ${idx}, and ${idx + 1}`);

        bumpDownTraceIndices(traceMap, idx+2);
        console.log(`Trace map after:`);
        printMap();
    }
}

function refreshConfigSelection(configHash) {
    return fetch(`/get_model_config?folder=${configHash}`)
        .then(response => response.json())
        .then(data => {
            if (data.config && data.checkpoints) {
                populateModelDetailsCard(data.config);

                var existing = configListGlobal.filter(c => c.hash === configHash);
                var maxRunIdx = existing?.reduce((maxValue, curr) => {
                        return Math.max(maxValue, curr.runIdx);
                    }, existing ? existing[0].runIdx : -1);

                console.log('maxRunIdx', maxRunIdx);
                var currentTrainIdx = maxRunIdx + 1;
                currentTrainLabel = `${configHash} ${maxRunIdx}`;

                configListGlobal.push({
                    'hash': configHash,
                    'checkpoints': existing?.checkpoints || [],
                    'trainingRun': [{
                        'train': [],
                        'eval': []
                    }],
                    'visible': true,
                    'showDropdown': maxRunIdx === 0,
                    'runIdx': currentTrainIdx
                });

                updateTraces(configHash, currentTrainIdx, true);
            }
        })
        .catch(error => console.error('Error:', error));
}

// Function to populate the model details card with config data
function populateModelDetailsCard(config) {
    var modelDetailsContainer = document.getElementById('modelDetails');

    modelDetailsContainer.innerHTML = '';

    // Start with common model details
    var commonDetailsHtml = `
        <div class="card-content">
            <div class="card-header">
                <h3>Model</h3>
            </div>
            <div class="label">Width</div><div class="value">${config.model.width}</div>
            <div class="label">Height</div><div class="value">${config.model.height}</div>
        </div>
        <div class="card-content">
            <div class="card-header">
                <h3>Dataset</h3>
            </div>
            <div class="label">Sample Rate</div><div class="value">${config.dataset.sample_rate}</div>
            <div class="label">Trunc len</div><div class="value">${config.dataset.trunc_len}</div>
            <div class="label"># FFT</div><div class="value">${config.dataset.n_fft}</div>
            <div class="label"># Samples</div><div class="value">${config.dataset.size}</div>
        </div>
        <div class="card-content">
            <div class="card-header">
                <h3>Training</h3>
            </div>
            <div class="label">Dropout</div><div class="value">${config.model.dropout}</div>
            <div class="label">Batch Size</div><div class="value">${config.batch_size}</div>
        </div>
    `;

    modelDetailsContainer.innerHTML += commonDetailsHtml;

    var architectureContainer = document.getElementById('architecture');

    var w = config.model.width;
    var h = config.model.height;

    // Now create the table for conv layer details
    var tableHtml = `
        <div>
            <div class="card-header">
                <h3>Conv Layers</h3>
            </div>
            <table class="conv-layers-table">
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Kernel Size</th>
                        <th>Maps</th>
                        <th>Stride</th>
                        <th>Pooling</th>
                    </tr>
                </thead>
                <tbody>
    `;

    config.model.conv_layers.forEach((layer, index) => {
        tableHtml += `
            <tr>
                <td>${index + 1}</td>
                <td>${layer.kernel_size}</td>
                <td>${layer.maps}</td>
                <td>${layer.stride}</td>
                <td>${layer.pool ? 'Yes' : 'No'}</td>
            </tr>
        `;
        w /= layer.stride;
        h /= layer.stride;
        if(layer.pool) {
            w /= 2;
            h /= 2;
        }
    });

    tableHtml += `</tbody></table></div>`;

    var numLayers = config.model.conv_layers.length;
    var finalSize = config.model.conv_layers[numLayers - 1].maps * w * h;
    var inputSize = finalSize;

    tableHtml += `
        <div>
            <div class="card-header" style="margin-top:10px">
                <h3>Linear Layers</h3>
            </div>
            <table class="conv-layers-table">
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Input Size</th>
                        <th>Output Size</th>
                        <th>Relu</th>
                        <th>Dropout</th>
                    </tr>
                </thead>
                <tbody>
    `;

    config.model.linear_layers.forEach((layer, index) => {
        tableHtml += `
            <tr>
                <td>${index + 1}</td>
                <td>${inputSize}</td>
                <td>${layer.out_size}</td>
                <td>${layer.relu ? "Yes" : "No"}</td>
                <td>${layer.dropout ? "Yes" : "No"}</td>
            </tr>
        `;
        inputSize = layer.out_size;
    });

    architectureContainer.innerHTML = tableHtml;
}

function onTrainingRunVisibilityChanged(hash, runIdx, checked) {
    console.log("Hash, runidx, checked:", hash, runIdx, checked);
    var elem = configListGlobal.find(e => e.hash === hash && e.runIdx === runIdx);
    if(elem) {
        elem.visible = checked;
    }
    updateTraces(hash, runIdx, checked);
}

function buildTrainingRunTable() {
    var configTable = document.getElementById('config-table');
    configTable.innerHTML = '';

    var table = document.createElement('table');
    table.className = 'conv-layers-table';
    var thead = table.createTHead();
    var row = thead.insertRow();
    var headers = ['Model', 'Checkpoint', 'Run', 'Plot'];
    headers.forEach(text => {
        var th = document.createElement('th');
        th.textContent = text;
        row.appendChild(th);
    });

    var tbody = table.createTBody();

    configListGlobal.forEach(data => {
        var hash = data.hash;
        var isChecked = data.visible;

        if(data.showDropdown) {
            var ckptDropdown = document.createElement('select');
            var checkpoints = data.checkpoints;

            checkpoints.sort();
            checkpoints.forEach(cp => {
                var ckptOption = document.createElement('option');
                ckptOption.value = cp.slice(0, -8);
                ckptOption.textContent = cp.slice(0, -8);
                ckptDropdown.appendChild(ckptOption);
            });
        }

        var visualize = document.createElement('input');
        visualize.type = "checkbox";
        visualize.id = `${hash}-${data.runIdx}`;
        visualize.checked = data.visible;
        visualize.addEventListener('change', function() {
            onTrainingRunVisibilityChanged(data.hash, data.runIdx, this.checked)
        });

        var row = tbody.insertRow();
        row.insertCell().textContent = data.showDropdown ? data.hash : '';
        if(data.showDropdown) {
            row.insertCell().appendChild(ckptDropdown);
        } else {
            row.insertCell();
        }
        row.insertCell().textContent = `${data.runIdx}`;
        row.insertCell().appendChild(visualize);
    });

    configTable.appendChild(table);
}

function fillConfigurations() {
    return fetch('/list_model_config_folders')
        .then(response => response.json())
        .then(configList => {
            configList.forEach(data => {
                var checkpoints = data.checkpoints;
                checkpoints = checkpoints.sort();
                var trainingRuns = data.config.training_runs || [{}];

                trainingRuns.forEach((tr, i) => {
                    configListGlobal.push({
                        'hash': data.hash,
                        'checkpoints': checkpoints,
                        'trainingRun': tr,
                        'visible': false,
                        'showDropdown': i === 0,
                        'runIdx': i
                    });
                });
            });
        })
        .catch(error => console.error('Error:', error));
}


// actions
function initialize() {
    createLossGraph();

    if (typeof(EventSource) !== "undefined") {
        var initSource = new EventSource("/initialize");

        initSource.addEventListener('load_progress', loadingListener, false);
        initSource.addEventListener('load_end', loadEndListener, false);
        initSource.addEventListener('refresh_config', function (event) {
            fillConfigurations()
            .then(() => refreshConfigSelection(event.data))
            .then(() => buildTrainingRunTable());
        }, false);
        initSource.onerror = function(error) {
            console.error("EventSource failed:", error);
            initSource.close();
        };
    } else {
        console.error("Your browser does not support server-sent events.");
    }
}

function startTraining() {
    if (typeof(EventSource) !== "undefined") {
        var trainSource = new EventSource("/train");

        trainSource.addEventListener('train_lifecycle', function(event) {
            var newData = event.data;
            document.getElementById("trainingProgressInfo").innerHTML = newData;
        }, false);

        trainSource.addEventListener('train_progress', trainingListener, false);
        trainSource.addEventListener('eval_progress', function(event) {
            var updateData = JSON.parse(event.data);
            document.getElementById("evalStatus").innerHTML = `Eval loss: ${updateData.loss.toFixed(4)}`;
        }, false);

        trainSource.addEventListener('checkpoint', checkpointSavedListener, false);
        trainSource.addEventListener('load_progress', loadingListener, false);
        trainSource.addEventListener('load_end', loadEndListener, false);
        trainSource.addEventListener('refresh_visuals', updateVizListener, false);
        trainSource.addEventListener('learning_rate', function(event) {
            var updateData = event.data;
            document.getElementById('learningRate').innerHTML = `${updateData}`;
        });

        trainSource.addEventListener('train_progress', updateTrainTrace, false);
        trainSource.addEventListener('eval_progress', updateEvalTrace, false);

        trainSource.onerror = function(error) {
            console.error("EventSource failed:", error);
            trainSource.close();
        };
    } else {
        console.error("Your browser does not support server-sent events.");
    }
}

function pauseTraining() {
    $.ajax({
        type: "POST",
        url: "/train/pause",
        success: function(response) {
            console.log("Training paused")
        }
    });
}

function listCheckpoints() {
    $.ajax({
        type: "GET",
        url: "/checkpoints",
        success: function(response) {
            console.log("Checkpoints: " + json.stringify(response))
        }
    });
}

function loadCheckpoint(filename) {
    $.ajax({
        type: "POST",
        url: "/checkpoints/load/" + filename,
        success: function(response) {
            console.log("Checkpoints: " + json.stringify(response))
        }
    });
}

function saveCheckpoint(filename) {
    $.ajax({
        type: "POST",
        url: "/checkpoints/" + filename,
        success: function(response) {
            console.log("Checkpoints: " + json.stringify(response))
        }
    });
}

window.onload = function() {
    initialize();
};
