import torch
import logging
import time
import torch.optim as optim
from torch import nn
from pathlib import Path
from torch.nn.utils import clip_grad_norm_

# logging setup
logger = logging.getLogger(__name__)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def train_one_epoch(
    model_type, model, dataloader, optimizer, criterion, scheduler, device=device
):
    """
    Train the model for one epoch.

    :param model_type: Type of model to train
    :type model_type: str
    :param model: Pytorch model to train
    :type model: chempleter.model.ChempleterModel
    :param dataloader: DataLoader containing training batches
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for updating model parameters (default: Adam)
    :type optimizer: torch.optim.Optimizer
    :param criterion: Loss function to compute training loss (default: CrossEntropyLoss)
    :type criterion: torch.nn.Module
    :param scheduler: Learning rate scheduler (default: ReduceLROnPlateau)
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device: Device to run training on (cpu or cuda)
    :type device: str
    :return: Average loss for the epoch
    :rtype: float
    """

    model.train()
    total_loss = 0

    for batch_idx, batch_tuple in enumerate(dataloader):
        # prepare batch
        batch = batch_tuple[0]
        batch_tensor_lengths = batch_tuple[1]
        batch = batch.to(device)

        # set inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:].clone()

        if model_type == "bridge":
            bridge_token_idx = 4 # default for [BRIDGE] token
            # ignore everything before bridge token for calcualting loss
            for token_sequence_idx in range(targets.size(0)):
                bridge_token_pos = (
                    inputs[token_sequence_idx] == bridge_token_idx
                ).nonzero(as_tuple=True)[0]  # find position of bridge token
                if len(bridge_token_pos) > 0:
                    targets[token_sequence_idx, : bridge_token_pos[0]] = (
                        0  # default padding index
                    )

        logits, _ = model(inputs, batch_tensor_lengths - 1)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        if batch_idx % 500 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)

    if scheduler:
        scheduler.step(avg_loss)

    return avg_loss


def start_training(
    n_epochs,
    model_type,
    model,
    dataloader,
    optimizer=None,
    criterion=None,
    scheduler=None,
    device=device,
    model_save_path=None,
    resume=False,
    checkpoint_path=None
):
    """
    Start training the model for a specified number of epochs.

    :param n_epochs: Number of epochs to train the model
    :type n_epochs: int
    :param model_type: Type of model to train
    :type model_type: str
    :param model: Type of model to train, either extend or bridge
    :type model_type: str
    :param model: Pytorch model to train or resume training
    :type model: chempleter.model.ChempleterModel
    :param dataloader: DataLoader containing training batches
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for updating model parameters (default: Adam)
    :type optimizer: torch.optim.Optimizer
    :param criterion: Loss function to compute training loss (default: CrossEntropyLoss)
    :type criterion: torch.nn.Module
    :param scheduler: Learning rate scheduler (default: ReduceLROnPlateau)
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device: Device to run training on (cpu or cuda)
    :type device: str
    :param model_save_path: Path to save the model checkpoint
    :type model_save_path: pathlib.Path
    :param resume: Flag for resuming training from a checkpoint
    :type resume: bool
    :param checkpoint_path: Path to a model checkpoint
    :type checkpoint_path: str
    """

    # get defaults
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not criterion:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    if not scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=3, factor=0.1
        )

    start_epoch = 0

    if resume is True:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path} to resume training.")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model state dict from checkpoint.")
            else:
                raise ValueError("model_state_dict not found in checkpoint file.")
            
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                logger.info(f"Loaded optimizer state dict from checkpoint and moved to device : {device}.")
                
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state dict from checkpoint.")
            
            start_epoch = checkpoint['epoch'] + 1 if "epoch" in checkpoint else 0
            last_loss = checkpoint['loss'] if "loss" in checkpoint else float("inf")
            logger.info(f"Set start_epoch : {start_epoch} and last_loss: {last_loss} from checkpoint.")

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.best = last_loss
                logger.info(f"Set scheduler best loss to last_loss: {last_loss} from checkpoint.")
        else:
            raise FileNotFoundError(f"checkpoint file not found at path: {str(checkpoint_path)}")
        

    # model to device
    model.to(device)
    if not model_save_path:
        model_save_path = Path().cwd()
    else:
        model_save_path = Path(model_save_path)

    current_lr = scheduler.get_last_lr()
    best_loss = last_loss if resume is True else float("inf")

    for epoch in range(start_epoch,n_epochs):
        start_time = time.time()
        avg_loss = train_one_epoch(
            model_type, model, dataloader, optimizer, criterion, scheduler, device
        )
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
        if current_lr != scheduler.get_last_lr():
            current_lr = scheduler.get_last_lr()
            print(f"Changed learning rate to : {current_lr}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "current_lr": current_lr,
                },
                model_save_path / "checkpoint.pt",
            )
            print(f"Saved model at Epoch {epoch}")

        print(f"Time taken for Epoch {epoch}: {time.time() - start_time}")
