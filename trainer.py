import math
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim import Optimizer

from typing import Tuple
import pytorch_lightning as L
import torch.nn.functional as F
from pydantic import BaseModel
from lightning.pytorch.loggers import TensorBoardLogger

from data.dataset import TokenDataset
from model.gpt2 import GPT2Model
import time

class GPT2Lighting(L.LightningModule):
    class Config(BaseModel):
        model: GPT2Model.Config = GPT2Model.Config()
        max_learning_rate: float = 6e-4
        min_learning_rate: float = 6e-4 * 0.1
        warmup_steps : int = 10
        max_steps : int = 50
        weight_decay : float = 0.1

    def __init__(self, config: "GPT2Lighting.Config"):
        super().__init__()

        self.model = GPT2Model(config.model)
        self.model = torch.compile(self.model)
        self.config = config
     
    def _create_scheduler(self, optimizer : Optimizer, config : 'GPT2Lighting.Config') -> LRScheduler:
        """Creates the custom learning rate scheduled optimizer.

        Args:
            optimizer (Optimizer): The optimizer that will have its learning rate changed

        Returns:
            LRScheduler
        """
                
        def _schedule(it : int) -> float:
            # to account for step = 0
            if it < config.warmup_steps:
                return config.max_learning_rate * (it+1) / config.warmup_steps
            
            if it > config.max_steps:
                return config.min_learning_rate
            
            decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 

            return config.min_learning_rate * coeff + (config.max_learning_rate - config.min_learning_rate)

        return LambdaLR(optimizer, lambda x : _schedule(x))

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        t0 = time.time()

        x, targets = batch
        logits = self.model(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.log("train_loss", loss)

        t1 = time.time()
        tokens_per_sec = len(batch) * x.shape[1] / (t1 - t0)

        self.log("tokens per second", tokens_per_sec)
        print(f'tokens per second  : {tokens_per_sec}')
        self.log("dt", (t1 - t0) * 1000)
        print(f'dt : {(t1 - t0) * 1000}')

        return loss

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return {'optimizer' : optimizer, 'lr_schedular' : {'scheduler' : self._create_scheduler(optimizer, self.config)}}


if __name__ == "__main__":
    model = GPT2Lighting(GPT2Lighting.Config())
    dataset = TokenDataset("data/input.txt")

    dataloader = DataLoader(dataset, batch_size=64, num_workers=16)
    logger = TensorBoardLogger("test_logs", name="testing_model")

    trainer = L.Trainer(
        precision='32',
        max_epochs=100, 
        logger=logger, 
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=5,
        )
    trainer.fit(model=model, train_dataloaders=dataloader)
