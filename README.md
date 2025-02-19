# Project Title: Optimizing Sentiment Analysis Models for Accurate Predictions

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Analysis Approach](#analysis-approach)
- [Key Findings](#key-findings)
- [How to run code](#how-to-run-code)
- [Technologies Used](#technologies-used)
- [Results & Visualizations](#results--visualizations)
- [Recommendation](#recommendation)
- [Contact](#contact)

## Overview

Sentiment analysis is a technique in natural language processing (NLP) and text mining that involves analyzing and determining the emotional tone or sentiment expressed in text. This project aims to evaluate and compare the performance of various word embeddings and sequence-to-sequence (seq2seq) models for sentiment analysis of restaurant reviews. The goal is to identify the best-performing models and configurations for analyzing customer sentiment.

## Dataset

- The dataset used in this analysis is composed of three files: [train.json](Data/train.json), [test.json](Data/test.json), and [val.json](Data/val.json).
- These files contain a total of 8,879 reviews for a single restaurant.
- Each review is categorized into one of the following eight aspects: food, service, staff, price, ambience, menu, place, and miscellaneous. The sentiment associated with each review is classified as positive, negative, or neutral.

## Objective

The objective of this project is to evaluate and compare the performance of various word embedding techniques and seq2seq models for sentiment analysis. The goal is to identify the best-performing combination for analyzing customer sentiment based on review text.

## Analysis Approach

Text Preprocessing & Data Preparation

- Clean review data by removing special characters, stop words, and irrelevant information.
- Prepare word embeddings:
  + GloVe: Uses pre-trained word vectors for better contextual understanding.
  + Word2Vec (CBOW & Skipgram): Captures semantic relationships between words.
  + FastText (CBOW & Skipgram): Enhances word representations, including out-of-vocabulary words.
- Prepare seq2seq models:
  + LSTM with Different Location Aspect: Incorporates location-based sentiment aspects.
  + LSTM with Attention: Implements an attention mechanism to focus on relevant text portions.
  + LSTM with Double Attention: Applies attention at both word and sentence levels.
- Ablation Study
  + RNN Model: Baseline model for comparison.
  + GRU Model: Alternative to LSTM with a simpler architecture and improved training efficiency.

## Key Findings

After evaluating multiple word embeddings and models, the best-performing combination for sentiment analysis was found to be Word2Vec Skipgram embeddings combined with LSTM with Attention mechanism. This combination provided:
- Superior word representation, capturing semantic relationships.
- Enhanced contextual understanding of customer reviews.
- Improved sentiment classification accuracy and interpretability.

## How to run code

1. Install Required Libraries: Ensure all necessary libraries such as pandas, matplotlib, seaborn, tensorflow, and gensim are installed like in the [file](Optimizing_Sentiment_Models.ipynb)
2. Load the Dataset: Import the dataset by loading the train.json, test.json, and val.json files.
3. Run the Analysis Notebooks: Execute the analysis notebooks in Jupyter to process the data, build and train the model, and visualize the results.

Run this process in Google Colab for easy execution and visualization.

## Technologies Used

- Python Code: Data processing and analysis were done in Python using libraries like pandas and numpy for data manipulation, gensim for Word2Vec and FastText embeddings, and nltk for text preprocessing tasks such as tokenization, stopwords removal, and stemming. Model evaluation was carried out with scikit-learn, and deep learning models were built and trained using torch.

- Visualization: For visualizing the results, matplotlib and seaborn were used for plotting, while wordcloud was utilized to generate word clouds to illustrate sentiment and aspect-wise insights.

## Results & Visualizations

![A-model1-same-embedding](https://github.com/user-attachments/assets/d532ecb1-5fe8-4b93-a87a-beef847f4a85)

Figure 1: Model Validation - LSTM with Different Location Aspect Using Word2Vec SkipGram Embeddings

![Amodel1-Gru-ablation](https://github.com/user-attachments/assets/70e9140c-13d5-485c-ab1e-123d2b948de2)

Figure 2: Model Validation - LSTM with Different Location Aspect Using NRR Ablation Study

![Amodel3-same-embedding](https://github.com/user-attachments/assets/70533fbb-c3e7-4549-80b0-d7db267ee810)

Figure 3: Model Validation - LSTM with Double Attention Using Word2Vec SkipGram Embeddings

![Amodel3-gru-ablation](https://github.com/user-attachments/assets/9067f28d-925f-48f3-9d46-0c1b905ecca1)

Figure 4: Model Validation - LSTM with Double Attention Using GRU Ablation Study

![Amodel2-RNN-ablation](https://github.com/user-attachments/assets/184fb6ec-f9e0-471b-bf6d-526c94b15a46)

Figure 5: Model Validation - LSTM with Attention Using NRR Ablation Study

![Abest-model](https://github.com/user-attachments/assets/23ddbe62-6f7d-4585-96a1-86c4479fad52)

Figure 6: Model Validation - LSTM with Attention Using Word2Vec SkipGram Embeddings (best model)

![image](https://github.com/user-attachments/assets/f9a84f07-81bc-4128-8fb2-d9b0b4e25b0a)
Figure 7: Training loss over epoches and training accuracy over epochs for best model

![image](https://github.com/user-attachments/assets/ed464599-229e-4699-926b-7d532a35eb17)
Figure 8: Attention visualization with aspect Staff using best model


![image](https://github.com/user-attachments/assets/906620a1-a65b-4bd7-b1a2-2acc7b0bb763)
Figure 9: Attention visualization with aspect Service using best model

**Noted:** For details report, please go to this [report]()

## Recommendation

To improve model, we can try:
- Use Transformer Models: Replace LSTM with BERT, RoBERTa, or DistilBERT for better context understanding.
- Enhance Word Embeddings: Fine-tune Word2Vec, FastText, or switch to pre-trained transformer embeddings.
- Fine-Tune Hyperparameters: Optimize learning rate, dropout, and batch size using Bayesian search.

## Contact

📧 Email: pearriperri@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/phan-chenh-6a7ba127a/) | Portfolio



