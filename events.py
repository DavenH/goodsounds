import json

def train_lifecycle_event(event_message):
    return f"event: train_lifecycle\ndata: {event_message}\n\n"

def train_progress_event(epoch: int, numEpochs: int, step: int, loss: float):
    data = json.dumps({'epoch': epoch, 'numEpochs': numEpochs, 'step': step, 'loss': loss})
    return f"event: train_progress\ndata: {data}\n\n"

def learning_rate_event(learning_rate):
    return f"event: learning_rate\ndata: {learning_rate}\n\n"

def eval_progress_event(epoch: int, numEpochs: int, step: int, eval_loss: float):
    data = json.dumps({'epoch': epoch, 'numEpochs': numEpochs, 'step': step, 'loss': eval_loss})
    return f"event: eval_progress\ndata: {data}\n\n"

def checkpoint_event(filename: str, epoch: int, step: int):
    data = json.dumps({'filename': filename, 'epoch': epoch, 'step': step})
    return f"event: checkpoint\ndata: {data}\n\n"

def refresh_visuals_event():
    return f"event: refresh_visuals\ndata: ''\n\n"

def refresh_config(config_hash):
    return f"event: refresh_config\ndata: {config_hash}\n\n"

def load_progress_event(index: int, size: int):
    data = json.dumps({'index': index, 'size': size})
    return f"event: load_progress\ndata: {data}\n\n"

def load_start_event(size: int):
    return f"event: load_start\ndata: {size}\n\n"

def load_end_event(size: int, elapsed_time: float):
    data = json.dumps({'size': size, 'elapsed': elapsed_time})
    return f"event: load_end\ndata: {data}\n\n"
