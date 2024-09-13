
# Fantasy Soccer News Analysis
## Introduction
Based on news articles, this project focuses on predicting the likelihood of a football player's chance of playing in the next game. By analyzing textual information, I aim to indicate if a player will be part of the main squad or be sidelined due to circumstances such as being injured or transferred. The project employs two deep learning models: a Recurrent Neural Network (RNN) using LSTM layers, and a Convolutional Neural Network (CNN). Both models are designed to extract relevant features from text and use them to predict player availability. 

## Configuration
The project was developed using Google Colab, which provides an easy-to-use environment for running Jupyter notebooks with GPU support. Below are the steps to recreate the development environment:

1. Clone the Repository: Clone this repository to your local machine opening your terminal and using git clone.
2. Visit Google Colab.
3. Upload the notebook file from this repository to Google Colab.
4. Setup and Run:
5.Ensure you have the necessary dependencies installed. You can install them using pip or conda as specified in the notebook.
Access:
6. Run the notebooks in Google Colab to execute the code and view results.

## Methodology
In this project, I followed a structured approach to text classification:

1. Data Preprocessing:

- Tokenization: The text data is tokenized into individual words or tokens.
- Filtering: Numeric tokens are filtered out.
- Vocabulary Building: Constructed a vocabulary from the processed text, assigning unique indices to each token.
- Embedding Layer: Used pre-trained GloVe embeddings (glove.6B.100d.txt) to initialize the embedding layer for better semantic representation of words in the dataset.
   
2. Model Implementation:
Two models are implemented to compare which one is better. 
- RNN Model: An Recurrent Neural Network model specifically an LSTM.
- CNN Model: A simple Convolutional Neural Network model as the baseline model.
3. Evaluation:
- Performance metrics and visualizations are used to evaluate and compare the models. (MSE, MAE and Loss Curves)
4. Results:
- Documented and compared to understand the strengths and weaknesses of each model.

## Blog Website
For more detailed insights and updates on this project, please visit the following link: [site](https://santiagoviteri01.github.io/santiagoviteri01.github.io./)


