from torch.utils.data import DataLoader

import torch
from vit_pytorch import ViT, MAE

from base_model import BaseModel
from events import eval_progress_event


class MaskedVit(BaseModel):
    def __init__(self, image_size: int, patch_size: int, n_categories: int, pretrain_dl: DataLoader):
        super(MaskedVit, self).__init__(n_categories)
        self.pretrain_dl = pretrain_dl
        self.config = dict(
            image_size=image_size,
            patch_size=patch_size,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            channels=1,
            decoder_dim=512,
            decoder_depth=6,
            masking_ratio=0.75
        )

        self.v = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = n_categories,
            dim = self.config["dim"],
            depth = self.config["depth"],
            heads = self.config["heads"],
            mlp_dim = self.config["mlp_dim"],
            channels = self.config["channels"]
        )

        self.mae = MAE(
            encoder = self.v,
            masking_ratio = self.config["masking_ratio"],  # the paper recommended 75% masked patches
            decoder_dim = self.config["decoder_dim"],      # paper showed good results with just 512
            decoder_depth = self.config["decoder_depth"]   # anywhere from 1 to 8
        )

    def pretrain(self,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 n_epochs: int,
                 folder: str
                 ):
        best_loss = 1000

        for epoch in range(n_epochs):
            step = 0
            for batch in self.pretrain_dl:
                optimizer.zero_grad()
                inputs = batch['spectrogram'].to(device)
                loss = self.mae(inputs)
                loss.backward()
                optimizer.step()
                best_loss = min(best_loss, loss.item())
                step += 1
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
                yield eval_progress_event(epoch, n_epochs, step * self.pretrain_dl.batch_size, loss.item())


        torch.save(self.v.state_dict(), f'{folder}/trained-vit-{best_loss:0.4f}.pt')

    def forward(self, x):
        return self.v(x)

    def get_config(self) -> dict:
        return self.config