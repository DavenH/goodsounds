
var currentSampleIdx = 0;

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

function refreshConfigSelection(folder) {
    fetch(`/get_model_config?folder=${folder}`)
        .then(response => response.json())
        .then(data => {
            if (data.config && data.checkpoints) {
                populateModelDetailsCard(data.config);

                var checkpointDropdown = document.getElementById('checkpointDropdown');
                checkpointDropdown.innerHTML = '';
                data.checkpoints.forEach(checkpoint => {
                    var option = document.createElement('option');
                    option.value = checkpoint;
                    option.textContent = checkpoint;
                    checkpointDropdown.appendChild(option);
                });
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
            <h3>Model</h3>
            <div class="label">Width</div><div class="value">${config.model.width}</div>
            <div class="label">Height</div><div class="value">${config.model.height}</div>
        </div>
        <div class="card-content">
            <h3>Dataset</h3>
            <div class="label">Sample Rate</div><div class="value">${config.dataset.sample_rate}</div>
            <div class="label">Trunc len</div><div class="value">${config.dataset.trunc_len}</div>
            <div class="label"># FFT</div><div class="value">${config.dataset.n_fft}</div>
            <div class="label"># Samples</div><div class="value">${config.dataset.size}</div>
        </div>
        <div class="card-content">
            <h3>Training</h3>
            <div class="label">Dropout</div><div class="value">${config.model.dropout}</div>
            <div class="label">Batch Size</div><div class="value">${config.batch_size}</div>
        </div>
    `;
    modelDetailsContainer.innerHTML += commonDetailsHtml;

    var architectureContainer = document.getElementById('architecture');

    // Now create the table for conv layer details
    var tableHtml = `
        <div>
            <h3>Conv Layers</h3>
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
    });

    tableHtml += `</tbody></table></div>`;

    architectureContainer.innerHTML = tableHtml;
}


function populateModelConfigDropdown() {
    var modelConfigDropdown = document.getElementById('modelConfigDropdown');
    modelConfigDropdown.innerHTML = '';

    fetch('/list_model_config_folders')
        .then(response => response.json())
        .then(folders => {
            folders.forEach(folder => {
                var option = document.createElement('option');
                option.value = folder;
                option.textContent = folder;
                modelConfigDropdown.appendChild(option);
            });
        })
        .catch(error => console.error('Error:', error));
}

// actions
function initialize() {

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

    var initialData = [{
        x: [],
        y: [],
        mode: 'lines',
        name: 'Train Loss'
    }, {
        x: [],
        y: [],
        mode: 'lines',
        name: 'Eval Loss'
    }];

    populateModelConfigDropdown();
    Plotly.newPlot('lossGraph', initialData, layout);

    if (typeof(EventSource) !== "undefined") {
        var initSource = new EventSource("/initialize");

        initSource.addEventListener('load_progress', loadingListener, false);
        initSource.addEventListener('load_end', loadEndListener, false);
        initSource.addEventListener('refresh_config', function (event) {
            refreshConfigSelection(event.data)
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
            document.getElementById("trainStatus").innerHTML = newData;
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

        trainSource.addEventListener('train_progress', function(event) {
            var updateData = JSON.parse(event.data);
            Plotly.extendTraces('lossGraph', {
                y: [[updateData.loss]],
                x: [[updateData.step]]
            }, [0]);
        }, false);

        trainSource.addEventListener('eval_progress', function(event) {
            var updateData = JSON.parse(event.data);
            Plotly.extendTraces('lossGraph', {
                y: [[updateData.loss]],
                x: [[updateData.step]]
            }, [1]);
        }, false);

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
