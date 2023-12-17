# Sentiment Analysis with Naive Bayes Classifier

This Python script performs sentiment analysis using a Naive Bayes classifier on a combined dataset of movie reviews from the NLTK and IMDb corpora. The script includes functions for data loading, preprocessing, model training, evaluation, metrics calculation, and making predictions. Additionally, it visualizes the sentiment distribution of example predictions.

## Prerequisites

Ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install nltk matplotlib
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/shan305/sentiment_analysis_-Naive_Bayes_Classifier.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd your-repository
   ```

3. **Run the Script:**

   ```bash
   python sentiment_analysis.py
   ```

4. **Review Results:**

   The script will train the Naive Bayes classifier, save the model, evaluate its performance, calculate metrics, make predictions on example sentences, and visualize the sentiment distribution.

## Script Overview

- **Dependencies:**
  - `nltk`: Natural Language Toolkit for natural language processing
  - `matplotlib`: Plotting library for data visualization

- **Features:**
  - Tokenization, stopwords removal, and feature extraction
  - Loading movie reviews from NLTK and IMDb datasets
  - Data preprocessing and splitting
  - Training a Naive Bayes classifier
  - Model evaluation, metrics calculation, and confusion matrix display
  - Saving and loading the trained classifier model
  - Making predictions on example sentences
  - Visualizing the sentiment distribution of predictions

- **Outputs:**
  - `metrics.txt`: Text file containing confusion matrix, precision, recall, and F1 score
  - `classifier_model.pickle`: Pickle file containing the trained Naive Bayes classifier
  - `sentiment_distribution.png`: Plot showing the predicted sentiment distribution for example sentences

## Customization

Feel free to customize the script according to your requirements. You can modify the example sentences, adjust the preprocessing steps, or explore different machine learning models for sentiment analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.