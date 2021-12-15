import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


def train(train_df, train_genre):
    models = []
    for i in range(1, 611):
        idx = train_df[train_df.userId == i].index
        min_idx = min(idx)
        max_idx = max(idx)

        x_train = train_genre.toarray()[min_idx:max_idx + 1]
        y_train = train_df.loc[idx].rating.values
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        models.append(model)

    return models

def make_submission(models, test_df, test_genre):

    pred = []

    for i in zip(test_df.userId.values.tolist(), test_genre.toarray()):
        idx, genre = i
        genre = genre.reshape(1, -1)
        model = models[idx - 1]
        cost = model.predict(genre)
        pred.append(cost)

    ans = pd.Series([i.item() for i in pred])
    submission.iloc[:, 1] = ans
    submission.to_csv("./submission.csv", index=False)


def recommend(models, userId: int, genre: str):
    model = models[userId - 1]
    genre_tfidf = Genre.transform([genre])
    rating = model.predict(genre_tfidf)

    print(f"userId: {userId}, Genre: {genre}, predict rating: {rating.item():.4f}")


if __name__ == "__main__":

    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    submission = pd.read_csv("./submission.csv")

    train_df['new_genre'] = train_df.genres.apply(lambda x: x.split('|')).apply(lambda x: " ".join(x))
    test_df['new_genre'] = train_df.genres.apply(lambda x: x.split('|')).apply(lambda x: " ".join(x))

    Genre = TfidfVectorizer().fit(train_df.new_genre)

    train_genre = Genre.transform(train_df.new_genre)
    test_genre = Genre.transform(test_df.new_genre)

    models = train(train_df, train_genre)

    make_submission(models, test_df, test_genre)

    recommend(models, 5, "Horror Thriller")




