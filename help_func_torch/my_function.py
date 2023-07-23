from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import torch
import matplotlib.pylab as plt
from PIL import Image
from copy import deepcopy
from tqdm import trange, tqdm


class MyDataset(Dataset):
    """This class return custom dataset object
        params: subset (input: dataframe)
                transform (input: transforms of torch)"""
    def __init__(self, directory=None, subset=None, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.path = directory
        self.le = LabelEncoder()
        self.labels_data = subset
        self.labels_data.loc[:, 'breed'] = self.le.fit_transform(self.labels_data['breed'])
        self.mapping = dict(enumerate(self.le.classes_))
        self.len = len(self.labels_data)

    def labels_dict(self):
        return self.mapping
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        self.x = Image.open(self.path + self.labels_data.iloc[index, 0] + '.jpg')
        if self.transform:
            self.x = self.transform(self.x)
        self.y = torch.tensor(self.labels_data.iloc[index, 1]).type(torch.LongTensor)
        return self.x, self.y
    

def train_model(model, criterion, optimizer, train, val, device, batch_size=32, epochs=10, patience_threshold=None, scheduler=None):
    """This function will train model and return history, best, latest model"""
    trainloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True)
    batches_per_epoch = len(trainloader)
    patience = 0

    latest = {'model': None, 'acc': 0, 'loss': 10e10, 'optimizer': None}
    best = {'model': None, 'acc': 0, 'loss': 10e10, 'optimizer': None}
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(epochs):
        with trange(batches_per_epoch, unit='batch') as pbar:
            pbar.set_description(f'{epoch + 1} epoch(s)')

            # train mode
            train_loss = 0
            train_acc = 0
            model.train()
            for x, y in trainloader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                _, label = torch.max(yhat, dim=-1)
                loss = criterion(yhat, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (label==y).sum().item()
                pbar.update()
                pbar.set_postfix_str(f'train_loss: {loss.item():.4f}')


            train_loss = round(train_loss/len(trainloader), 4)
            train_acc = round(train_acc/len(train), 3)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # eval mode
            val_loss = 0
            val_acc = 0
            model.eval()
            for x, y in valloader:
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                _, label = torch.max(yhat, dim=-1)
                val_loss += criterion(yhat, y).item()
                val_acc += (label==y).sum().item()

            val_loss = round(val_loss/len(valloader), 4)
            val_acc = round(val_acc/len(val), 3)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            
            # update learning rate
            if scheduler:
                pbar.set_postfix_str(f'train_loss: {train_loss} train_acc: {train_acc} val_loss: {val_loss} val_acc: {val_acc} learning_rate: {scheduler.get_last_lr()[0]:.0e}')
                scheduler.step()
            else:
                pbar.set_postfix_str(f'train_loss: {train_loss} train_acc: {train_acc} val_loss: {val_loss} val_acc: {val_acc}')

            # best model and early stop
            if val_loss < best['loss']:
                best['model'] = deepcopy(model)
                best['acc'] = deepcopy(val_acc)
                best['loss'] = deepcopy(val_loss)
                best['optimizer'] = deepcopy(optimizer)
                patience = 0


            latest['model'] = deepcopy(model)
            latest['acc'] = deepcopy(val_acc)
            latest['loss'] = deepcopy(val_loss)
            latest['optimizer'] = deepcopy(optimizer)
            if patience_threshold:
                patience += 1
                if patience == patience_threshold:
                    break

    return history, best, latest


def test_model(model, test_data, device):
    """This function will test best or lastest model from train_model functon
       And return y_true, y_pred"""
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for x, y in tqdm(iter(test_data), desc='Time remain', total=len(test_data)):
        x, y = x.to(device), y.to(device)
        y_p = F.softmax(model(x.unsqueeze(0)), dim=-1)
        y_prob.append(y_p.tolist()[0])
        y_true.append(y.item())
        y_pred.append(y_p.argmax().item())

    return y_true, y_pred, y_prob


def plot_history(history):
    "This function wil plot history of loss and acc according to epochs"
    plt.figure(figsize=(20, 10)) 
    plt.tight_layout()

    plt.subplot(1, 2, 1); plt.plot(history['train_loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss'); plt.legend()

    plt.subplot(1, 2, 2); plt.plot(history['train_acc'], label='accuracy')
    plt.plot(history['val_acc'], label='val_accuracy'); plt.legend()
