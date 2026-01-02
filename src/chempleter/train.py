import torch
import time
import torch.optim as optim
from torch import nn
from pathlib import Path
from torch.nn.utils import clip_grad_norm_

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def train_one_epoch(model_type,model, dataloader, optimizer, criterion, scheduler, device=device):
    """
    Train the model for one epoch.

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

        if model_type=="bridge":
            bridge_token_idx = 4 # default for ["BRIDGE"]
            # ignore everything before bridge token for calcualting loss
            for token_sequence_idx in range(targets.size(0)):
                bridge_token_pos = (inputs[token_sequence_idx] == bridge_token_idx).nonzero(as_tuple=True)[0] # find position of bridge token
                if len(bridge_token_pos)>0:
                    targets[token_sequence_idx, :bridge_token_pos[0]] = 0 # default padding index

        
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
):
    """
    Start training the model for a specified number of epochs.

    :param n_epochs: Number of epochs to train the model
    :type n_epochs: int
    :param model: Type of model to train, either extend or bridge
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
    :param model_save_path: Path to save the model checkpoint
    :type model_save_path: pathlib.Path
    """
    # model to device
    model.to(device)

    # get defaults
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not criterion:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    if not scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=3, factor=0.1
        )
    if not model_save_path:
        model_save_path = Path().cwd()
    else:
        model_save_path = Path(model_save_path)

    current_lr = scheduler.get_last_lr()
    best_loss = float("inf")
    for epoch in range(n_epochs):
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

