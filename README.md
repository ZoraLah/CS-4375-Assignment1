# CS4375 Assignment 1 — Sentiment Analysis with Neural Networks
This project creates two models (Feedforward Neural Network (FFNN) and Recurrent Neural Network (RNN)) that performs a 5-class sentiment analysis based on Yelp reviews.

## Necessary files:  
- `ffnn.py` — Feedforward Neural Network model
- `rnn.py` — Recurrent Neural Network model
- `Data_Embedding/` — contains larger training, validation, and test JSON files
- `word_embedding.pkl` — pretrained word embeddings (required for RNN)

## Requirements
- Python 3.8.x
- Install dependencies:
```
pip install numpy torch tqdm matplotlib
```

## How to Run

**FFNN:**
```
python ffnn.py --hidden_dim [hidden_dim parameter]  --epochs [epoch_parameter] --train_data Data_Embedding/training.json --val_data Data_Embedding/validation.json
```

**RNN:**
```
python rnn.py --hidden_dim [hidden_dim parameter]  --epochs [epoch_parameter] --train_data Data_Embedding/training.json --val_data Data_Embedding/validation.json
```

After training is completed, a learning curve plot will be automatically created and saved into the current directory (`[model]_hd[hidden_dim parameter]_learning_curve.png`)

## Note
`word_embedding.pkl` and the `Data_Embedding/` folder are not included in this 
repository due to file size limits. Please download them from eLearning and place 
them in the root directory before running.