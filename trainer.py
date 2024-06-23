import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Tuple
import pytorch_lightning as L
import torch.nn.functional as F
from pydantic import BaseModel
from lightning.pytorch.loggers import TensorBoardLogger

from data.dataset import TokenDataset
from model.gpt2 import GPT2Model

class GPT2Lighting(L.LightningModule):

    class Config(BaseModel):
        model : GPT2Model.Config = GPT2Model.Config()
        learning_rate : float = 0.001

    def __init__(self, config : 'GPT2Lighting.Config'):
        super().__init__()
        
        self.model = GPT2Model(config.model)
        self.config = config
        
    def training_step(self, batch : Tuple[Tensor, Tensor], batch_idx : int):
        
        x, targets = batch
        logits = self.model(x)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        self.log("train_loss", loss)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        return optimizer
    
    
if __name__ == "__main__":
    
    model = GPT2Lighting(GPT2Lighting.Config())
    dataset = TokenDataset('data/input.txt')
    
    dataloader = DataLoader(dataset, batch_size=6, num_workers=16)
    logger = TensorBoardLogger('test_logs', name='testing_model')
    
    trainer = L.Trainer(max_epochs=10, logger=logger)
    trainer.fit(model=model, train_dataloaders=dataloader)