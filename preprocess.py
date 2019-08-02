import numpy as np
import pandas as pd
import re
from collections import defaultdict

def reduce_and_split(dataset, proportion, train_proportion, val_proportion, test_proportion):
    # Reduce the dataset's size and all the preprocess of reviews.
    by_sentiment = defaultdict(list)

    for _, row in dataset.iterrows():
        by_sentiment[row.sentiment].append(row.to_dict())

    reviews_subset = []
    for _, item_list in by_sentiment.items():
        np.random.shuffle(item_list)
        len_reviews = len(item_list)
        len_sub_reviews = int(len_reviews * proportion)

        reviews_subset.extend(item_list[:len_sub_reviews])
    subset_dataset = pd.DataFrame(reviews_subset)

    # Split the dataset into train, validation, test datasets
    by_sentiment = defaultdict(list)
    for _, row in subset_dataset.iterrows():
        by_sentiment[row.sentiment].append(row.to_dict())

    subset_list = []
    for _, item_list in by_sentiment.items():
        np.random.shuffle(item_list)

        n_total = len(item_list)
        n_train = int(train_proportion * n_total)
        n_val = int(val_proportion * n_total)
        n_test = int(test_proportion * n_total)

        for item in item_list[:n_train]:
            item['split'] = 'train'

        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'

        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
        subset_list.extend(item_list)


    subset_arr = np.array(subset_list)
    np.random.shuffle(subset_arr)

    train_review, train_sentiment, val_review, val_sentiment, test_review, test_sentiment = seperate_datasts(subset_arr)

    # Preprocess text
    train_review = [preprocess_text(review) for review in train_review]
    val_review = [preprocess_text(review) for review in val_review]
    test_review = [preprocess_text(review) for review in test_review]

    # Transform sentiment from 'postive' to 1 / from 'negative' to 0
    train_sentiment = transform_sentiment2index(train_sentiment)
    val_sentiment = transform_sentiment2index(val_sentiment)
    test_sentiment = transform_sentiment2index(test_sentiment)

    return train_review, train_sentiment, val_review, val_sentiment, test_review, test_sentiment

def seperate_datasts(subset_arr):
    train_review = []
    train_sentiment = []

    val_review = []
    val_sentiment = []

    test_review = []
    test_sentiment = []

    for data in subset_arr:
        if data['split'] == 'train':
            train_review.append(data['review'])
            train_sentiment.append(data['sentiment'])

        if data['split'] == 'val':
            val_review.append(data['review'])
            val_sentiment.append(data['sentiment'])

        if data['split'] == 'test':
            test_review.append(data['review'])
            test_sentiment.append(data['sentiment'])

    return train_review, train_sentiment, val_review, val_sentiment, test_review, test_sentiment


def transform_sentiment2index(sentiment_data):
    return np.array([1 if sentiment == 'positive' else 0 for sentiment in sentiment_data])


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
