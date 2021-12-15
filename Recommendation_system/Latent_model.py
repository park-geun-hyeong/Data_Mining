import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

def train(rank):
    print(f"rank : {rank}")

    movie = torch.randn(num_movie + 1, rank, requires_grad=True)
    user = torch.randn(num_user + 1, rank, requires_grad=True)

    optimizer = optim.Adam([movie, user], lr=0.01)
    loss_fn = nn.MSELoss()

    for e in range(1000):

        train_loss = 0
        optimizer.zero_grad()

        pred = torch.sum(movie[items] * user[users], dim=1)
        cost = loss_fn(pred, ratings)
        loss = cost + 0.001 * torch.sum(movie ** 2) + 0.001 * torch.sum(user ** 2)

        loss.backward()
        optimizer.step()
        train_loss += loss

        val_loss = 0
        with torch.no_grad():
            val_pred = torch.sum(movie[items_valid] * user[users_valid], dim=1)
            loss = loss_fn(val_pred, ratings_valid)
            val_loss += loss

        if e % 100 == 0:
            print(f"epoch : {e + 1}, train_loss: {train_loss:.4f}, valid_loss: {val_loss:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    submission = pd.read_csv("./submission.csv")

    Train_df, Valid_df = train_test_split(train_df, test_size=0.1)

    Train_df = Train_df.reset_index(drop=True)
    Valid_df = Valid_df.reset_index(drop=True)

    items = torch.LongTensor(Train_df['movieId'].values)
    users = torch.LongTensor(Train_df['userId'].values)
    ratings = torch.FloatTensor(Train_df.rating.values)

    items_valid = torch.LongTensor(Valid_df['movieId'].values)
    users_valid = torch.LongTensor(Valid_df['userId'].values)
    ratings_valid = torch.FloatTensor(Valid_df.rating.values)

    num_user = int(np.max(Train_df.userId.values))
    num_movie = int(np.max(Train_df.movieId.values))

    rank = [5, 10, 15, 20]

    for r in rank:
        train(rank = r)
