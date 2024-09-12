
# Fantasy Soccer News Analysis
## Introduction
This project focuses on predicting the likelihood of a football player's chance of playing in the next game, based on news articles and player updates. By analyzing textual information, we aim to predict if a player will participate or be sidelined due to injury or transfer-related circumstances. The project employs two deep learning models: a Recurrent Neural Network (RNN) using LSTM layers, and a Convolutional Neural Network (CNN), both built with PyTorch Lightning.

The core task involves extracting relevant features from text, such as injury reports, and using them to predict player availability. The dataset is derived from news headlines and player-specific data, such as injuries and game predictions.

## Data Processing
Data Collection and Preprocessing: Cleaned the data, removed irrelevant tokens, applied stemming and tokenization using nltk_tokenizer.
Vocabulary Building: Constructed a vocabulary from the processed text, assigning unique indices to each token.
Embedding Layer: Used pre-trained GloVe embeddings (glove.6B.100d.txt) to initialize the embedding layer for better semantic representation of words in the dataset.
## Models
LSTM Model
Architecture: Implemented a Long Short-Term Memory (LSTM) model for sequence prediction. The model used GloVe embeddings, and the LSTM layer captures temporal dependencies in the text, such as a player's injury history.
Training: The model was trained with mean squared error (MSE) as the loss function and Adam optimizer, with early stopping and checkpointing for saving the best model.
Evaluation: Validation and test sets were used to assess model performance, logging Mean Squared Error (MSE) and Mean Absolute Error (MAE) during training.
CNN Model
Architecture: Implemented a CNN for text classification. The model applies multiple convolutional filters to the word embeddings, capturing local features such as phrases and word patterns that may indicate injury risk.
Training: Similar to the LSTM model, the CNN was trained using MSE loss and Adam optimizer, with early stopping based on validation performance.
## Model Evaluation
Metrics: The models' performance was evaluated using MSE and MAE. Loss values during training were logged and plotted to observe convergence.
Testing: Predictions were made using news phrases to estimate the probability of a player's involvement in the next game.
Example Predictions
Used the trained model to predict the probability of a player playing based on news headlines like "Chelsea agree transfer, player on loan to Reading" or "Ankle injury, no date for return."
Key Libraries and Tools
PyTorch Lightning: Framework for training the models.
NLTK: Tokenization, stemming, and vocabulary building.
Scikit-learn: Data splitting and evaluation metrics.
GloVe Embeddings: Pre-trained word embeddings for capturing semantic relationships in text data.
