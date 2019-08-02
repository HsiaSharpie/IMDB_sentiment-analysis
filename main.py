import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from argparse import Namespace

from preprocess import reduce_and_split
from dataset import IMDB_Dataset
from vocab_and_tokenize import Vocabulary, Tokenize
from model import Sentiment_Classifier


args = Namespace(
    raw_dataset_csv = "/Users/samhsia/Desktop/github/Deep_Learning-Practice/IMDB_sentiment/IMDB_Dataset.csv",
    proportion = 0.1,
    train_proportion=0.8,
    val_proportion=0.1,
    test_proportion=0.1,
    seed = 111,
    final_csv="data/IMDB/reviews_with_splits.csv"
)

def main():
    raw_dataset = pd.read_csv(args.raw_dataset_csv)
    train_review, train_sentiment, val_review, val_sentiment, test_review, test_sentiment = reduce_and_split(raw_dataset, args.proportion, args.train_proportion, args.val_proportion, args.test_proportion)

    # For training in batches, we set equal sequence length
    seq_length = 300
    train_sentences = Tokenize(train_review, seq_length) # return numpy.ndarray with shape ([train_size, seq_length])
    val_sentences = Tokenize(val_review, seq_length)
    test_sentences = Tokenize(test_review, seq_length)

    # Convert the inputs datatype from ndarray to torch.tensor
    train_data = TensorDataset(torch.from_numpy(train_sentences.input_seq), torch.from_numpy(train_sentiment))
    val_data = TensorDataset(torch.from_numpy(val_sentences.input_seq), torch.from_numpy(val_sentiment))
    test_data = TensorDataset(torch.from_numpy(test_sentences.input_seq), torch.from_numpy(test_sentiment))

    # Use the DataLoader to split the Dataset with batch_size
    batch_size = 50

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # Parameters to set the model
    vocab_size = len(train_sentences.vocab2idx) + 1
    output_size = 1
    embedding_dim = 300
    hidden_dim = 128
    n_layers = 2

    # check whether to use cuda or not
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Sentiment_Classifier(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    model.to(device)

    # Set the model's Hyper-Parameters
    lr = 0.005
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 1
    counter = 0
    clip = 5
    valid_loss_min = np.Inf

    # training_process
    model.train()
    for i in range(epochs):
        train_hidden = model.init_hidden(batch_size)

        for inputs, labels in train_loader:
            counter += 1
            train_hidden = tuple([element.data for element in train_hidden])
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()
            output, train_hidden = model(inputs, train_hidden)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % 1 == 0:
                val_hidden = model.init_hidden(batch_size)
                val_losses = []
                model.eval()

                for inputs, labels in val_loader:
                    val_hidden = tuple([element.data for element in val_hidden])
                    inputs, labels = inputs.to(device), labels.to(device)
                    output, val_hidden = model(inputs, val_hidden)

                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                    valid_loss_min = np.mean(val_losses)

    # testing -> Calculate accuracy
    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) #rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))

if __name__ == "__main__":
    main()
