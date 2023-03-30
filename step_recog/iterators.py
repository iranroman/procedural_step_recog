from step_recog.models import GRUNet
import torch
from torch import nn
import numpy as np


def train(train_loader, val_loader, learn_rate=0.001, hidden_dim=256, EPOCHS=5, model_type="GRU", output_dim = 34, n_layers=2):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Setting common hyperparameters
    batch_size, _, input_dim = next(iter(train_loader))[0][0].shape
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(1024,2304,input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    print("Starting Training of {} model".format(model_type))
    # Start training loop
    best_val_loss = float('inf')
    for epoch in range(1,EPOCHS+1):
        model.train()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            label = nn.functional.one_hot(label,output_dim)
            counter += 1
            if model_type == "GRU":
                #h = h.data
                h = torch.zeros_like(h)
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model((x[0].to(device).float(),x[1].to(device).float()), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Training Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        print("Epoch {}/{} Done, Total training Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))

        model.eval()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in val_loader:
            label = nn.functional.one_hot(label,output_dim)
            counter += 1
            if model_type == "GRU":
                #h = h.data
                h = torch.zeros_like(h)
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model((x[0].to(device).float(),x[1].to(device).float()), h)
            loss = criterion(out, label.to(device).float())
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Validation Loss for Epoch: {}".format(epoch, counter, len(val_loader), avg_loss/counter))
        val_loss = avg_loss/len(val_loader)
        print("\t\t\tEpoch {}/{} Done, Total validation Loss: {}".format(epoch, EPOCHS, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("\t\t\tFound new best validation loss (saving model)")
            torch.save(model.state_dict(),'model_best.pt')
    return model


def evaluate(model, val_loader):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.eval()
    outputs = []
    targets = []
    for x, label in val_loader:
        h = model.init_hidden(x.shape[0])
        out, h = model((x[0].to(device).float(),x[1].to(device)), h)
        outputs.append(np.argmax(out.cpu().detach().numpy(),axis=1))
        targets.append(label.numpy())
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    np.save('outputs.npy',outputs)
    np.save('targets.npy',targets)
    return outputs, targets
