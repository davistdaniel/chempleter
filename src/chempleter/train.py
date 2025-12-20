
import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def train_one_epoch(model,dataloader,optimizer,criterion,device):
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

        # if batch_idx % 100 == 0:
        #     print(f"Batch {batch_idx + 1}/{len(dataloader)} "
        #       f"- Loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)


    return avg_loss

def start_training(n_epochs,model,dataloader, optimizer, criterion, device=device):
    model.to(device)

    best_loss = float("inf")
    for epoch in range(n_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, "checkpoint.pt")
            print(f"Saved model at Epoch {epoch}")