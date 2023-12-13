import hashlib
import json
import os.path
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cnn import ConvNetModel
from dataset import BaseAudioDataset
from events import train_progress_event, checkpoint_event, eval_progress_event, refresh_visuals_event
from subset_sampler import SubsetSampler


def save_checkpoint(
        model: ConvNetModel,
        train_set: BaseAudioDataset,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        state: dict,
        batch_size: int,
        ds_class_name: str,
        epoch: int,
        step: int,
        avg_eval_loss: float):
    model_state = {
        'dataset': ds_class_name,
        'epoch': epoch,
        'step': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    config_hash, config, folder = get_config_maybe_creating_folder(model, train_set, batch_size, ds_class_name)
    config["training_runs"] = state["training_runs"][config_hash]

    filename = f"{folder}/epoch_{epoch:03}_step_{step}_loss_{avg_eval_loss:.6f}.pth.tar"
    print(f"Saving checkpoint {filename}")

    with open(f"{folder}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saving checkpoint {filename}")

    yield checkpoint_event(filename, epoch, step)

    torch.save(model_state, filename)


def get_config_maybe_creating_folder(
        model: ConvNetModel,
        train_set: BaseAudioDataset,
        batch_size: int,
        ds_class_name: str):
    config_no_training_runs = dict(
        model=model.get_config(),
        dataset=train_set.get_config(),
        batch_size=batch_size
    )

    hash_str = f"{hashlib.md5(json.dumps(config_no_training_runs, sort_keys=True).encode('utf-8')).hexdigest()}"
    config_hash = hash_str[4:10]
    folder = f"checkpoints/{ds_class_name}/{config_hash}"

    config_file_name = os.path.join(folder, f"config.json")

    if not os.path.exists(folder):
        os.mkdir(folder)

    if os.path.exists(config_file_name):
        with open(config_file_name, 'r') as f:
            config = json.load(f)
    else:
        with open(config_file_name, 'w') as f:
            json.dump(config_no_training_runs, f, indent=2)
            config = config_no_training_runs

    if "training_runs" not in config:
        config["training_runs"] = []
    return config_hash, config, folder


def load_checkpoint(filename,
                    model: torch.nn.Module,
                    optim: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch'], checkpoint['step']


def train(
        max_subset_samples: int,
        device: torch.device,
        epochs: int,
        batch_size: int,
        ds_class_name: str,
        state: dict
):
    best_eval_loss = 1e9
    perf_stagnation = 0
    torch.manual_seed(0)

    model: ConvNetModel = state["model"]
    optimizer: torch.optim.Optimizer = state["optimizer"]
    train_set: BaseAudioDataset = state["train_set"]
    eval_set: BaseAudioDataset = state["eval_set"]
    start_epoch = state.get("start_epoch", 0)
    step = state.get("step", 0)
    state["paused"] = False

    config_hash, config, _ = get_config_maybe_creating_folder(model, train_set, batch_size, ds_class_name)

    sampler = SubsetSampler(train_set, max_subset_samples)
    if "sampler" in state:
        sampler.load_state_dict(state["sampler"])
    else:
        yield from sampler.refresh(state)
        state["sampler"] = sampler.save_state_dict()

    train_dl = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    eval_dl = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    loss_fn = train_set.get_loss_function().to(device)

    # when our loss curve plateaus, there's almost always more room for
    # improvement by reducing the learning rate.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    train_run = dict(eval=[], train=[])

    if config_hash not in state["training_runs"]:
        state["training_runs"][config_hash] = config["training_runs"]

    state["training_runs"][config_hash].append(train_run)

    # epochs = 0
    for epoch in range(start_epoch, epochs):
        if state["paused"]:
            state["start_epoch"] = epoch
            state["step"] = step
            state["scheduler"] = scheduler.state_dict()
            break

        if epoch % 3 == 0:
            yield refresh_visuals_event()

        if epoch % 1 == 0:
            print(f"Evaluating epoch {epoch}...")

            model.eval()
            accum_eval_loss = 0.0
            # turning off the gradient calculations speeds up inference
            with torch.no_grad():
                for batch in eval_dl:
                    with state["mutex"]:
                        inputs = batch['spectrogram'].to(device)
                        labels = batch['label'].to(device)
                        outputs = model(inputs)

                        loss = loss_fn(outputs, labels)
                        accum_eval_loss += loss.item()

            avg_eval_loss = accum_eval_loss / len(eval_dl)
            print(f"Epoch: {epoch}, step {step}, eval_loss: {avg_eval_loss:3.3}")
            # avg_eval_acc = accum_accuracy / len(eval_dl)
            scheduler.step(avg_eval_loss)
            yield eval_progress_event(epoch, epochs, step * batch_size, avg_eval_loss)
            train_run["eval"].append((step * batch_size, avg_eval_loss))

            if epoch >= 10 and epoch % 10 == 0:
                yield from save_checkpoint(model, train_set, optimizer, scheduler, state, batch_size,
                                           ds_class_name, epoch, step, avg_eval_loss)

            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
            else:
                print(f"Performance stagnation: {perf_stagnation}")
                perf_stagnation += 1

        model.train()

        for batch in train_dl:
            if state["paused"]:
                break

            with state["mutex"]:
                # gradients in the optimizer are still from the last batch - zero them
                optimizer.zero_grad()

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
                optimizer.step()

                step += 1

            # if count % 10 == 0:
            print(f"Epoch: {epoch}, step {step}, loss: {loss.item():3.3}")
            yield train_progress_event(epoch, epochs, step * batch_size, loss.item())

            train_run["train"].append((step * batch_size, loss.item()))
        # accum_accuracy = 0.0

        # # load a new random selection of training data into memory
        # if epoch > 0 and epoch % 100 == 0:

        if perf_stagnation > 10:
            perf_stagnation = 0
            print(f"Performance has stagnated at epoch {epoch}, refreshing data.")
            yield from sampler.refresh(state)
            state["sampler"] = sampler.save_state_dict()
