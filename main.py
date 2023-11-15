import random

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from cnn import ConvNetModel
from dataset import GoodSoundsDataset

# Apply the dataset class
root_path = '/home/daven/data/audio/good-sounds'
width, height = 256, 128
truncate_len = 65536
batch_size = 128
epochs = 20

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
    root_path = sys.argv[1]
    dataset = GoodSoundsDataset(root_path, truncate_len, width, height)
    model = ConvNetModel(width, height, len(dataset.categories)).to(device)
    optim = optim.Adam(model.parameters())

    # when our loss curve plateaus, there's almost always more room for
    # improvement by reducing the learning rate.
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.7, patience=8, verbose=True)

    loss_fn = nn.CrossEntropyLoss().to(device)
    torch.manual_seed(0)

    train_set, eval_set = torch.utils.data.random_split(dataset, [15308, 1000])

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_dl = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    # writer = SummaryWriter()

    start_epoch = 0
    count = 0
    best_eval_loss = 1e9
    # start_epoch, count = load_checkpoint('checkpoints/epoch_249_step_31748_loss_0.00013662.pth.tar')

    for epoch in range(start_epoch, epochs):
        # puts the dropout and batchnorm components into correct mode
        model.train()

        for batch in train_dl:
            # gradients in the optimizer are still from the last batch - zero them
            optim.zero_grad()

            inputs = batch['spectrogram'].to(device)
            labels = batch['index'].to(device)

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

            if count % 10 == 0:
                print(f"Epoch: {epoch}, step {count}, loss: {loss.item():3.3}")

        print(f"Evaluating epoch {epoch}...")

        model.eval()
        accum_eval_loss = 0.0
        accum_accuracy = 0.0

        # turning off the gradient calculations speeds up inference
        with torch.no_grad():
            for batch in eval_dl:
                inputs = batch['spectrogram'].to(device)
                labels = batch['index'].to(device)
                outputs = model(inputs)
                pred_idcs = outputs.argmax(dim=1)

                # calculate categorization accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accum_accuracy += correct / labels.size(0)

                loss = loss_fn(outputs, labels)
                accum_eval_loss += loss.item()

        avg_eval_loss = accum_eval_loss / len(eval_dl)
        avg_eval_acc = accum_accuracy / len(eval_dl)
        scheduler.step(avg_eval_loss)

        print(f"Eval loss: {avg_eval_loss:.6}, eval accuracy: {100*avg_eval_acc:3.2f}%")

        # checkpoint if it's an improvement
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            filename = f"checkpoints/t65k/epoch_{epoch}_step_{count}_loss_{avg_eval_loss:.6f}.pth.tar"
            print(f"Saving checkpoint {filename}...")

            save_checkpoint({
                'epoch': epoch,
                'step': count,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
            }, filename=filename)

        if avg_eval_acc > 0.995:
            print("Accuracy threshold reached, stopping early")
            break

    random_ind = random.sample(range(len(eval_set)),12)
    random_samples = [eval_set[i] for i in random_ind]

    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=[f"Sample {i+1}" for i in range(12)],
        horizontal_spacing=0.015,
        vertical_spacing=0.05
    )

    for i, sample in enumerate(random_samples):
        data, true_label = (sample['spectrogram'], sample['metadata']['label'])

        # Generate prediction and calculate loss
        model.eval()

        with torch.no_grad():
            # we 'unsqueeze' here to simulate having a batch size of 1,
            # as the model expects some batch dimension
            input = sample['spectrogram'].unsqueeze(0).to(device)
            label = torch.tensor(sample['index']).unsqueeze(0).to(device)
            output = model(input)

            # return the index of the maximum value of the output
            pred_idx = output.argmax(dim=1)
            pred_label = dataset.categories[pred_idx.item()]

        # Add subplot
        row, col = (i // 4) + 1, (i % 4) + 1
        fig.add_trace(go.Heatmap(z=data.cpu().squeeze(0).numpy(), colorscale='Viridis'), row=row, col=col)
        fig.layout.annotations[i].update(text=f'Pred: {pred_label}, Label: {true_label}')

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.show()
