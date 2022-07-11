import datetime
import torch
import torch.nn as nn


def training_loop(
    epochs, model, optimizer, loss_fn,
    train_loader, val_loader,
    device=None
):
    # Set device for model and input tensors.
    if not device:
        device = (torch.device('mps') if torch.backends.mps.is_available()
                else torch.device.device('cpu'))

    # Make sure model is on the specified device.
    if model.device != device:
        model.to(device=device)
    
    # Training loop.
    for epoch in range(1, epochs+1):
        time_start = datetime.datetime.now()
        loss_train = 0.0

        for images, labels in train_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            # Deactivate gradients.
            with torch.no_grad():
                val_loss = 0.0
                for features, labels in val_loader:
                    features = features.to(device=device)
                    labels = labels.to(device=device)
                    outputs = model(features)
                    loss = loss_fn(outputs)

                    val_loss += loss.item()

        time_elapsed = datetime.datetime.now() - time_start
        print(f'Epoch: {epoch}, \
            Average Time per Batch: {time_elapsed / len(train_loader)}, \
            Training Loss: {loss_train / len(train_loader)}, \
            Validation Loss: {val_loss / len(val_loader)}')
