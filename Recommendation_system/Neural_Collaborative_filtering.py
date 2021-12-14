import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Model
class NET(nn.Module):
    def __init__(self, user_num, item_num, emb_num, dropout):
        '''

        :param user_num: unique user number
        :param item_num: unique item number
        :param emb_num: embedding size
        :param dropout: dropout probability
        '''

        super(NET, self).__init__()

        self.dropout = dropout
        self.embed_user = nn.Embedding(user_num, emb_num)
        self.embed_item = nn.Embedding(item_num, emb_num)

        ## fc layer
        self.fc_layers = \
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),

                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
            )

        ## predict layer
        self.predict_layer = nn.Linear(16, 1)

        ## weight initialize
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        interaction = torch.cat((embed_user, embed_item), -1)
        output = self.fc_layers(interaction)
        concat = output

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

## DataSset
class dataset(Dataset):
    def __init__(self, df):
        '''
        :param df: Dataframe for Dataset
        '''

        super(dataset, self).__init__()
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        username = torch.LongTensor(self.df.userId.values)[idx].unsqueeze(0)
        movie = torch.LongTensor(self.df.movieId.values)[idx].unsqueeze(0)
        rating = torch.FloatTensor(self.df.rating.values)[idx]

        return username.to(device), movie.to(device), rating.to(device)

def train(model, epochs):

    '''

    :param model: model for training(NET)
    :param epochs: epochs for training
    :return: Trained Model

    print training state( epoch, train loss, valid loss)
    '''
    print("Training Start!\n")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for data in train_loader:
            user, movie, rating = data

            optimizer.zero_grad()

            pred = model(user, movie)
            loss = loss_fn(pred, rating)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss = []
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                user, movie, rating = data

                pred = model(user, movie)
                loss = loss_fn(pred, rating)

                val_loss.append(loss.item())

        lr_sc.step(np.mean(val_loss))

        print(f"epoch : {epoch + 1}, train_loss: {np.mean(train_loss):.4f}, val_loss: {np.mean(val_loss):.4f}")

def make_submission(model, test_df):
    '''

    :param model: Trained Model for Testing
    :param test_df: Dataframe for prediction
    :return: None

    just make submission csv file to your path

    '''
    print()
    print("Prediction Start!")
    test_df['rating'] = 0
    test_dataset = dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    pred = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            u, m, _ = data
            pr = model(u, m).clamp(0, 5)
            pred.append(pr.detach().cpu().numpy())

    pred = np.concatenate(pred)
    submission['rating'] = pred
    submission.to_csv('./new_submission.csv', index=False)

    print("Prediction End!")

if __name__ == "__main__":

    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    submission = pd.read_csv("./submission.csv")

    num_user = int(np.max(train_df.userId.values)) + 1
    num_movie = int(np.max(train_df.movieId.values)) + 1

    # Split Data for validation
    Train_df, Valid_df = train_test_split(train_df, test_size=0.1)

    Train_df = Train_df.reset_index(drop=True)
    Valid_df = Valid_df.reset_index(drop=True)


    # Dataset, DataLoader
    train_dataset = dataset(Train_df)
    val_dataset = dataset(Valid_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    ## Model
    model = NET(user_num=num_user, item_num=num_movie, emb_num=128, dropout=0.3)

    ## Optimizer,  loss function, lr_scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    lr_sc = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    ## Train
    train(model = model, epochs = 10)

    ## Prediction
    make_submission(test_df)


