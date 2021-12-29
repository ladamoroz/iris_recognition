import torch


def train(model, train_dataloader, epochs, optimizer, base_loss_function, aux_loss_function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            base_out, aux_out = model(inputs)

            base_loss = base_loss_function(base_out, labels)
            aux_loss = aux_loss_function(aux_out, inputs)

            loss = base_loss + aux_loss
            loss.backward()

            optimizer.step()

    return model