import torch

def train(model, train_loader,  epochs, device, learning_rate, momentum, best_model_path, first_epoch=1):

    loss_file = 'train/loss.csv'
    if first_epoch == 1:
        file = open(loss_file,'w+')
        file.write('epoch,loss\n')
        file.close()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_loss = 100
    
    for epoch in range(first_epoch, epochs + 1):
        
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
            
        file = open(loss_file,'a+')
        file.write(str(epoch)+','+str(total_loss)+'\n')
        file.close()    
        print("Epoch: {}, CELoss: {:f}".format(epoch, total_loss /len(train_loader)))
        if(total_loss < best_loss):
            best_loss = total_loss
            torch.save(model.state_dict(), best_model_path)
        