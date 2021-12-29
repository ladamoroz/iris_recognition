import torch

def test(model, test_dataloader, base_loss_function, aux_loss_function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        base_out, aux_out = model(inputs)

        base_loss = base_loss_function(base_out, labels)
        aux_loss = aux_loss_function(aux_out, inputs)

        loss = base_loss + aux_loss
        total_loss += loss.item()
    return total_loss