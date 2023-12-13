import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_balanced_accuracy(dataloader, model, num_classes, device=None):
    """Evaluate accuracy of a model on the given data set."""
    truep_dict = np.zeros((num_classes,), dtype=int)
    label_dict = np.zeros((num_classes,), dtype=int)
    for X, y in dataloader:
        # If device is the GPU, copy the data to the GPU.
        X, y = X.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            y = y.long()
            preds = torch.argmax(model(X), dim=1)
            truep_dict, label_dict = balanced_accuracy(preds, y, truep_dict, label_dict)
            # print(truep_dict, label_dict)

    recall_dict = truep_dict / label_dict
    # print(recall_dict.mean())
    return recall_dict.mean()


def balanced_accuracy(preds, label, truep_dict, label_dict):
    preds_np = preds.cpu().detach().numpy()
    label_np = label.cpu().detach().numpy()
    for i in range(preds_np.shape[0]):
        label_dict[label_np[i]] += 1
        if preds_np[i] == label_np[i]:
            truep_dict[preds_np[i]] += 1
    return truep_dict, label_dict


def train_model(model, num_classes, lr, criterion, train_iter, test_iter, num_epochs=100):
    """Train and evaluate a model with CPU or GPU."""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)
    if torch.cuda.is_available():
        print('training on', device)
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        # train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        truep_train_dict, label_train_dict = np.zeros((num_classes,), dtype=int), np.zeros((num_classes,), dtype=int)
        n, start = 0, time.time()
        for X, y in train_iter:
            model.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                truep_train_dict, label_train_dict = balanced_accuracy(torch.argmax(y_hat, dim=1), y, truep_train_dict,
                                                                       label_train_dict)
                n += y.shape[0]

        test_acc = evaluate_balanced_accuracy(test_iter, model, num_classes, device)

        train_loss_epoch = train_l_sum / n
        train_acc_epoch = (truep_train_dict / label_train_dict).mean()

        val_acc_epoch = evaluate_balanced_accuracy(test_iter, model, num_classes, device)

        train_loss.append(train_loss_epoch.item())
        train_acc.append(train_acc_epoch.item())
        val_acc.append(val_acc_epoch)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_epoch, test_acc,
                 time.time() - start))
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'checkpoint/efficientB0_model.pt')
    torch.save(model.state_dict(), 'checkpoint/efficientB0_model.pt')
    return train_loss, train_acc, val_acc