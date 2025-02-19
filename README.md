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
- 



-------------------------------------------


## Project Overview:

Dataset: train.json, test.json, and val.json

Restaurant dataset that comprises 8,879 reviews of a single restaurant. These reviews are categorized into eight distinct aspects: food, service, staff, price, ambience, menu, place, and miscellaneous.

Sentiment analysis is a technique in natural language processing (NLP) and text mining that involves analyzing and determining the emotional tone or sentiment expressed in a piece of text. The goal is to understand the feelings, opinions, or attitudes conveyed by the text, whether it's positive, negative, neutral, or even more specific emotions like anger, joy, or sadness.

## Objectives:

Evaluate and compare the performance of various word embeddings and seq2seq models for sentiment analysis. The aim is to identify the best-performing models and configurations for analyzing customer sentiment.

## Project Structure:

Outline the models and embeddings being tested: GloVe, Word2Vec (CBOW and Skipgram), FastText (CBOW and Skipgram), and Seq2Seq models (LSTM with different configurations), as well as the RNN and GRU models in the ablation study

1. Text Preprocessing & Data Preparation:
- Clean the review data by removing noise such as special characters, stop words, and any irrelevant information to prepare the text for analysis.
2. Prepare word embeddings:
- GloVe: Description of the GloVe word embeddings and how they are utilized in the model.
- Word2Vec (CBOW and Skipgram): Explanation of the Continuous Bag of Words (CBOW) and Skipgram models, highlighting their differences and how they represent word relationships.
- FastText (CBOW and Skipgram): Introduction to FastText, its ability to represent out-of-vocabulary words, and how it improves upon Word2Vec.
3. Prepare seq2seq models:
- LSTM with Different Location Aspect: Description of the LSTM architecture, focusing on the addition of location-based aspects.
- LSTM with Attention: Explanation of the attention mechanism in LSTM to focus on relevant parts of the sequence while making predictions.
- LSTM with Double Attention: Introduce the concept of double attention, where attention mechanisms are applied at both the word and sentence levels to improve sentiment understanding.
4. Ablation Study:
- RNN Model: A basic Recurrent Neural Network model to establish a baseline for comparison.
- GRU Model: Description of the GRU model and how it differs from LSTM, focusing on its simpler architecture and potential efficiency in training.

For more detail information, you can have a look at report file and jupyter notebook file on colab.

---

## Outcome

After evaluating multiple word embeddings and models, the best-performing combination for sentiment analysis was found to be Word2Vec Skipgram embeddings combined with the LSTM model with Attention mechanism. This model outperformed others in terms of sentiment classification accuracy and interpretability.

Word2Vec Skipgram embeddings provided superior word representation, especially for capturing semantic relationships between words, which enhanced the model's ability to understand context in customer reviews.
The LSTM with Attention model further improved performance by allowing the model to focus on important parts of the review text, such as keywords related to aspects like "food," "service," or "ambience," thereby improving the overall sentiment prediction.

Model Performance: 

![image](https://github.com/user-attachments/assets/f9a84f07-81bc-4128-8fb2-d9b0b4e25b0a)
Figure 1: Training loss over epoches and training accuracy over epochs

Confusion matrix & classification report:
![confusion_matrix](https://github.com/user-attachments/assets/641ee56c-85fe-44c2-8620-716eef466edf)

![image](https://github.com/user-attachments/assets/ed464599-229e-4699-926b-7d532a35eb17)
Figure 2: Attention visualization with aspect Staff


![image](https://github.com/user-attachments/assets/906620a1-a65b-4bd7-b1a2-2acc7b0bb763)
Figure 3: Attention visualization with aspect Service


