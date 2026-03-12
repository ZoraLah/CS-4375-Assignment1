import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt # using for analysis section to make graphs


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h 
        self.W1 = nn.Linear(input_dim, h) 
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5 # The output dimension is 5 since we have 5 classes (extremely positive, positive, neutral, negative, extremely negative)
        self.W2 = nn.Linear(h, self.output_dim) 
        self.softmax = nn.LogSoftmax(dim=0) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        h1 = self.W1(input_vector)
        h1 = self.activation(h1)
        
        # [to fill] obtain output layer representation
        output = self.W2(h1)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    
    # tracking metrics for analysis section
    training_accuracies = []
    validation_accuracies = []  
    training_losses = []
    
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # loss = None --> removed to track training loss 
        correct = 0
        total = 0
        epoch_loss = 0.0 # tracking epoch loss 
        loss_count = 0 # tracking number of losses for average calculation 
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            epoch_loss += loss.item() # tracking loss for analysis section
            loss_count += 1 # have to track number of losses for average calculation
            loss.backward()
            optimizer.step()
            
        # updating metric tracking for training!
        training_accuracy = correct / total
        average_loss = epoch_loss / loss_count
        training_accuracies.append(training_accuracy)
        training_losses.append(average_loss)
            
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        
        # added this to check if average training loss is working
        print("Average training loss for epoch {}: {:.4f}".format(epoch + 1, average_loss)) 

        # loss = None --> removed to track validation loss 
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        with torch.no_grad(): # don't need to compute gradients during validation
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                # loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    # example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    # if loss is None:
                    #     loss = example_loss
                    # else:
                    #     loss += example_loss
                # loss = loss / minibatch_size
                
            # updating metric tracking for validation!
            validation_accuracy = correct / total
            validation_accuracies.append(validation_accuracy)
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {:.4f}".format(epoch + 1, correct / total))
            print("Validation time for this epoch: {:.2f}".format(time.time() - start_time))

    # write out to results/test.out --> making a plot learning curve
    epochs_range = range(1, args.epochs + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color_loss)
    ax1.plot(epochs_range, training_losses, color=color_loss, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    
    color_training_acc = 'tab:blue'
    color_validation_acc = 'tab:green'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Accuracy')  
    ax2.plot(epochs_range, training_accuracies, color=color_training_acc, label='Training Accuracy')
    ax2.plot(epochs_range, validation_accuracies, color=color_validation_acc, label='Validation Accuracy')
    ax2.tick_params(axis='y')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('FFNN Learning Curve (hidden_dim={})'.format(args.hidden_dim))
    fig.tight_layout()
    
    plot_filename = 'ffnn_learning_curve_hidden_dim_{}.png'.format(args.hidden_dim)
    plt.savefig(plot_filename)
    plt.show()