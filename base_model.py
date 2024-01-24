import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, n_categories: int):
        super(BaseModel, self).__init__()
        self.visualize_outputs = False
        self.n_categories = n_categories
        self.intermediate_outputs = []

    def get_config(self) -> dict:
        raise NotImplementedError()