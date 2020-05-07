'''
loss='crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
'''
import torch

def train(model, train_loader, epochs, device):
    optimizer = optim.Adadelta(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.to(device)
        model.train()

        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            model.zero_grad()

            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        print("Epoch: {}, CELoss: {}".format(epoch, total_loss / len(train_loader)))
