import json
import os.path
import random
from base64 import b64encode

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from cnn import ConvNetModel
from dataset import GoodSounds, FSDKaggle2019
from subset_sampler import SubsetSampler

# config
width, height = 512, 128
truncate_len = 60000
sample_rate = 16000
batch_size = 1024
epochs = 500
max_subset_samples = 8000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['step']

if __name__ == "__main__":
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
        data_per_wav = 15 * sample_rate // truncate_len
        train_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/train_noisy_post_competition.csv',
                                  root_path + '/FSDKaggle2019.audio_train_noisy',
                                  sample_rate, truncate_len, data_per_wav, width, height)
        # train_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/train_curated_post_competition.csv',
        #                           root_path + '/FSDKaggle2019.audio_train_curated',
        #                           sample_rate, truncate_len, data_per_wav, width, height)
        eval_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/test_post_competition.csv',
                                  root_path + '/FSDKaggle2019.audio_test',
                                  sample_rate, truncate_len, data_per_wav, width, height)
    else:
        raise ModuleNotFoundError


    model = ConvNetModel(width, height, len(eval_set.categories)).to(device)
    optim = optim.Adam(model.parameters())
    loss_fn = eval_set.get_loss_function().to(device)
    # train_set.preload()
    eval_set.preload()

    # when our loss curve plateaus, there's almost always more room for
    # improvement by reducing the learning rate.
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=8, verbose=True)
    # scheduler = CosineAnnealingLR(optim, 20)
    torch.manual_seed(0)

    sampler = SubsetSampler(train_set, max_subset_samples)
    train_dl = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    eval_dl = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    best_eval_loss = 1e9
    perf_stagnation = 0
    start_epoch = 0
    count = 0
    # start_epoch, count = load_checkpoint("checkpoints/FSDKaggle2019/QyMzYw/epoch_20_step_210_loss_0.071580.pth.tar")
    # epochs = 0
    for epoch in range(start_epoch, epochs):
        # puts the dropout and batchnorm components into correct mode
        model.train()

        for batch in train_dl:
            # gradients in the optimizer are still from the last batch - zero them
            optim.zero_grad()

            inputs = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)

            # predict our outputs with the model
            outputs = model(inputs)

            # calculate the loss (error) - how well our model did in matching the labels
            loss = loss_fn(outputs, labels)

            # calculate the loss surface -- if you could tweak every parameter in the model slightly, for each one,
            # which way makes the loss go down.
            loss.backward()

            # take a small step in that direction that makes the loss go down
            optim.step()

            count += 1

            # if count % 10 == 0:
            print(f"Epoch: {epoch}, step {count}, loss: {loss.item():3.3}")
        # accum_accuracy = 0.0

        if epoch % 5 == 0:
            print(f"Evaluating epoch {epoch}...")

            model.eval()
            accum_eval_loss = 0.0
            # turning off the gradient calculations speeds up inference
            with torch.no_grad():
                for batch in eval_dl:
                    inputs = batch['spectrogram'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(inputs)

                    # pred_idcs = outputs.argmax(dim=1)
                    #
                    # # calculate categorization accuracy
                    # _, predicted = torch.max(outputs.data, 1)
                    # correct = (predicted == labels).sum().item()
                    # accum_accuracy += correct / labels.size(0)

                    loss = loss_fn(outputs, labels)
                    accum_eval_loss += loss.item()

            avg_eval_loss = accum_eval_loss / len(eval_dl)
            # avg_eval_acc = accum_accuracy / len(eval_dl)
            scheduler.step(avg_eval_loss)

            # print(f"Eval loss: {avg_eval_loss:.6}, eval accuracy: {100*avg_eval_acc:3.2f}%")
            print(f"Eval loss: {avg_eval_loss:.6}")

            # checkpoint if it's an improvement
            if avg_eval_loss < best_eval_loss * 0.995:
                best_eval_loss = avg_eval_loss

                config = dict(model=model.get_config(), dataset=train_set.get_config(), batch_size=batch_size)
                hash_str = f"{hash(json.dumps(config, sort_keys=True))}"
                config_hash = str(b64encode(hash_str.encode('utf-8')))[4:10]
                folder = f"checkpoints/{ds_class_name}/{config_hash}"

                config_file_name = os.path.join(folder, f"config.json")

                if not os.path.exists(folder):
                    os.mkdir(folder)

                if not os.path.exists(config_file_name):
                    with open(config_file_name, 'w') as f:
                        json.dump(config, f, indent=2)

                filename = f"{folder}/epoch_{epoch}_step_{count}_loss_{avg_eval_loss:.6f}.pth.tar"
                print(f"Saving checkpoint {filename}")

                save_checkpoint({
                    'dataset': ds_class_name,
                    'epoch': epoch,
                    'step': count,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),
                }, filename=filename)
            else:
                perf_stagnation += 1

        if epoch > 0 and epoch % 25 == 0:
            sampler.refresh()

        if perf_stagnation > 50:
            break

        # if avg_eval_acc > 0.995:
        #     print("Accuracy threshold reached, stopping early")
        #     break

    n_rows = 3
    n_cols = 2

    random_ind = random.sample(range(len(eval_set)), n_rows * n_cols)
    random_samples = [eval_set[i] for i in random_ind]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols * 2, # 1 plot for image, 1 for bar chart
        subplot_titles=[f"Sample {i+1}" for i in range(n_rows * n_cols * 2)],
        horizontal_spacing=0.015,
        vertical_spacing=0.05
    )

    with torch.no_grad():
        model.eval()

        for row in range(n_rows):
            for col in range(n_cols):
                image_idx = row * n_cols + col
                sample = random_samples[image_idx]
                data, true_label = (sample['spectrogram'], sample['metadata']['label'])

                # Generate prediction and calculate loss
                # we 'unsqueeze' here to simulate having a batch size of 1,
                # as the model expects some batch dimension
                input = sample['spectrogram'].unsqueeze(0).to(device)
                output = model(input)

                pred_weights = torch.sigmoid_(output)
                topk_values, topk_indices = torch.topk(pred_weights.cpu().squeeze(0), 10)

                topk_values = topk_values.flip(dims=[0])
                topk_indices = topk_indices.flip(dims=[0])

                # Create subset of categories based on top-10 indices
                top_categories = [eval_set.categories[i] for i in topk_indices.tolist()]

                # Add subplot
                fig.add_trace(go.Bar(y=top_categories, x=topk_values.numpy(), orientation='h'), row=row+1, col=2 * col + 1)
                fig.layout.annotations[2 * image_idx].update(text=f'Predictions')
                fig.add_trace(go.Heatmap(z=data.cpu().squeeze(0).numpy(), colorscale='Viridis'), row=row+1, col=2 * col + 2)
                fig.layout.annotations[2 * image_idx + 1].update(text=f'Label: {true_label}')

                # the fraction of the window corresponding with the x-domain of this pair of charts
                leftmost = (2 * col + 0) / (2 * n_cols)
                rightmost = (2 * col + 2) / (2 * n_cols)
                predictions_domain = [leftmost, leftmost * 0.8 + rightmost * 0.2]
                spectrogram_domain = [leftmost * 0.8 + rightmost * 0.2, leftmost * 0.2 + rightmost * 0.8]
                fig.update_xaxes(domain=predictions_domain, row=row + 1, col=2 * col + 1)
                fig.update_xaxes(domain=spectrogram_domain, row=row + 1, col=2 * col + 2)

    # fig.update_layout(
    #     margin=dict(l=10, r=10, t=30, b=10)
    # )
    fig.show()
