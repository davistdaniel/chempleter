import torch
import torch.optim as optim
from torch import nn
from pathlib import Path

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device=device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits, _ = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)

        if batch_idx % 500 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

    if scheduler:
        scheduler.step(avg_loss)

    return avg_loss


def start_training(
    n_epochs,
    model,
    dataloader,
    optimizer=None,
    criterion=None,
    scheduler=None,
    device=device,
    model_save_path=None,
):
    model.to(device)

    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not criterion:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    if not scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=3, factor=0.1
        )
        current_lr = scheduler.get_last_lr()
    if not model_save_path:
        model_save_path = Path().cwd()
    else:
        model_save_path = Path(model_save_path)

    best_loss = float("inf")
    for epoch in range(n_epochs):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, scheduler, device
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
                    "loss": avg_loss,
                    "current_lr": current_lr,
                },
                model_save_path / "checkpoint.pt",
            )
            print(f"Saved model at Epoch {epoch}")
