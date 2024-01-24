import json
import os
from threading import Lock

import plotly
import torch
import torch.optim as optim
from flask import Flask, jsonify, render_template, Response, stream_with_context, send_from_directory, request
from torch.utils.data import DataLoader

import trainer
from dataset import GoodSounds, FSDKaggle2019
from events import train_lifecycle_event, refresh_visuals_event, refresh_config
from vit import MaskedVit

# config
width, height = 256, 256
truncate_len = 128000
sample_rate = 32000
batch_size = 256
epochs = 500
max_subset_samples = 5000
lr = 2e-4
train_set, eval_set = None, None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from figures import create_predictions_figure

state = dict(
    train_set=None,
    eval_set=None,
    model=None,
    optimizer=None,
    paused=False,
    training_runs=dict(),
    mutex = Lock()
)
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  # The HTML template should have the buttons and audio element.

@app.route('/initialize')
def initialize():
    def init_generator():
        import sys
        print(sys.argv)

        # first parameter is root directory of dataset
        root_path = sys.argv[1]

        # second parameter is name of dataset
        ds_class_name = sys.argv[2]

        if ds_class_name == "GoodSounds":
            dataset = GoodSounds(root_path, sample_rate, truncate_len, width, height)
            train_amt = round(0.8 * len(dataset))
            train_set, eval_set = torch.utils.data.random_split(dataset, [train_amt, len(dataset) - train_amt])

        elif ds_class_name == "FSDKaggle2019":
            data_per_wav = 1
            # train_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/train_noisy_post_competition.csv',
            #                           root_path + '/FSDKaggle2019.audio_train_noisy',
            #                           sample_rate, truncate_len, data_per_wav, width, height)
            train_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/train_curated_post_competition.csv',
                                      root_path + '/FSDKaggle2019.audio_train_curated',
                                      sample_rate, truncate_len, data_per_wav, width, height)
            eval_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/test_post_competition.csv',
                                     root_path + '/FSDKaggle2019.audio_test',
                                     sample_rate, truncate_len, data_per_wav, width, height)
            state["audio_path"] = root_path + '/FSDKaggle2019.audio_test'
        else:
            raise ModuleNotFoundError

        # model = ConvNetModel(width, height, len(eval_set.categories)).to(device)

        pretrain_dl = DataLoader(train_set, batch_size=batch_size)

        model = MaskedVit(width, 32, len(eval_set.categories), pretrain_dl).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(optimizer.param_groups[0]['lr'])
        state["model"] = model
        state["optimizer"] = optimizer
        state["train_set"] = train_set
        state["eval_set"] = eval_set

        config_hash, config, folder = trainer.get_config_maybe_creating_folder(model, train_set, batch_size, ds_class_name)

        yield refresh_config(config_hash)
        yield refresh_visuals_event()
        yield from eval_set.preload(state)
        yield from model.pretrain(optimizer, 10, folder)

    return Response(stream_with_context(init_generator()), mimetype='text/event-stream')


@app.route('/train')
def train():
    def train_generator():

        import sys
        # second parameter is name of dataset
        ds_class_name = sys.argv[2]

        yield train_lifecycle_event("Training has (re)started")
        yield from trainer.train(
            max_subset_samples=max_subset_samples,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            ds_class_name=ds_class_name,
            state=state)
        yield train_lifecycle_event("Training has finished / paused")

    return Response(stream_with_context(train_generator()), mimetype='text/event-stream')


@app.route('/train/pause', methods=['POST'])
def pause_training():
    state["paused"] = True
    return Response()


@app.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """
    :return: a list of *.pth.tar checkpoint file names
    """
    checkpoint_dir = "checkpoints/FSDKaggle2019"

    f = []
    for (dirpath, dirnames, filenames) in os.walk(checkpoint_dir):
        print(dirpath, dirnames)
        f.extend(filenames)

    return jsonify(f)


@app.route('/checkpoints/load/<checkpoint>', methods=['POST'])
def load_checkpoint(checkpoint):
    checkpoint_path = f"checkpoints/FSDKaggle2019/{checkpoint}"
    if not os.path.isfile(checkpoint_path):
        return jsonify({"error": "Checkpoint file not found."}), 404

    start_epoch, count = trainer.load_checkpoint(
        checkpoint_path,
        state["model"],
        state["optimizer"],
        state["scheduler"]
    )
    state["start_epoch"] = start_epoch
    state["step"] = count

    return jsonify({"message": "Checkpoint loaded successfully.", "start_epoch": start_epoch, "step": count})


@app.route('/get_model_config', methods=['GET'])
def get_model_config():
    print('Getting model config')
    folder = request.args.get('folder')
    config_path = os.path.join('checkpoints/FSDKaggle2019', folder, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        checkpoints = [file for file in os.listdir(os.path.join('checkpoints/FSDKaggle2019', folder))
                       if file.endswith('.pth.tar')]
        return jsonify({'config': config, 'checkpoints': checkpoints})
    else:
        return jsonify({'error': 'Folder not found'}), 404


@app.route('/list_model_config_folders', methods=['GET'])
def list_model_config_folders():
    checkpoint_dir = "checkpoints/FSDKaggle2019"

    config_list = []
    for (dirpath, dirnames, filenames) in os.walk(checkpoint_dir):
        for folder in dirnames:
            config_path = os.path.join(checkpoint_dir, folder, 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                checkpoints = [file for file in os.listdir(os.path.join(checkpoint_dir, folder))
                               if file.endswith('.pth.tar')]
                config_list.append({
                    'hash': folder,
                    'config': config,
                    'checkpoints': checkpoints
                })
        break
    return jsonify(config_list)


@app.route('/evaluate_sample/<index>', methods=['POST'])
def evaluate_sample(index):
    print("Generating figure")
    if state["model"] is None:
        print("Not initialized...")
        return jsonify({'figure': '', 'audio': ''})
    else:
        fig, fig2, audio_path = create_predictions_figure(state, int(index), device)
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        kernel_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({'spectrogram': graph_json, 'intermediates': kernel_json, 'audio': audio_path})


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    audio_directory = state["audio_path"]
    return send_from_directory(audio_directory, filename)

app.run(debug=True, threaded=True)