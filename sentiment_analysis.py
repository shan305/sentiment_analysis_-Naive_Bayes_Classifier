import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
import matplotlib.pyplot as plt
import random
import os
import pickle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Feature Extraction
def extract_features(words):
    return dict([(word, True) for word in words])

# Load Movie Reviews Dataset
def load_movie_reviews():
    nltk_positive_reviews = [(list(movie_reviews.words(fileids=[f])), 'pos') for f in movie_reviews.fileids('pos')]
    nltk_negative_reviews = [(list(movie_reviews.words(fileids=[f])), 'neg') for f in movie_reviews.fileids('neg')]

    # IMDb dataset
    imdb_positive_reviews = []
    imdb_negative_reviews = []

    # Download IMDb dataset
    from nltk.corpus import PlaintextCorpusReader
    imdb_path = nltk.data.find('corpora/movie_reviews')
    imdb_corpus = PlaintextCorpusReader(imdb_path, '.*\.txt')
    
    for fileid in imdb_corpus.fileids():
        words = imdb_corpus.words(fileid)
        sentiment = fileid.split('/')[0]
        if sentiment == 'pos':
            imdb_positive_reviews.append((words, 'pos'))
        elif sentiment == 'neg':
            imdb_negative_reviews.append((words, 'neg'))

    # Combine reviews
    reviews = nltk_positive_reviews + nltk_negative_reviews + imdb_positive_reviews + imdb_negative_reviews
    random.shuffle(reviews)

    return reviews

# Preprocess Data
def preprocess_data(reviews):
    stop_words = set(stopwords.words('english'))
    processed_reviews = []

    for words, sentiment in reviews:
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        processed_reviews.append((words, sentiment))

    return processed_reviews

# Split Data
def split_data(processed_reviews, split_ratio=0.8):
    split = int(len(processed_reviews) * split_ratio)
    train_data, test_data = processed_reviews[:split], processed_reviews[split:]
    return train_data, test_data

# Train Naive Bayes Classifier
def train_classifier(train_data):
    training_features = [(extract_features(words), sentiment) for words, sentiment in train_data]
    classifier = NaiveBayesClassifier.train(training_features)
    return classifier

# Evaluate Model
def evaluate_model(classifier, test_data):
    test_features = [(extract_features(words), sentiment) for words, sentiment in test_data]
    accuracy = nltk_accuracy(classifier, test_features)
    print(f'Accuracy: {accuracy:.2%}')

# Calculate Metrics and Display Confusion Matrix
def calculate_metrics(classifier, test_data):
    test_features = [(extract_features(words), sentiment) for words, sentiment in test_data]
    test_labels = [sentiment for _, sentiment in test_data]
    predicted_labels = [classifier.classify(features) for features, _ in test_features]

    cm = ConfusionMatrix(test_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    precision = cm['pos', 'pos'] / (cm['pos', 'pos'] + cm['neg', 'pos'])
    recall = cm['pos', 'pos'] / (cm['pos', 'pos'] + cm['pos', 'neg'])
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision:.2%}')
    print(f'Recall: {recall:.2%}')
    print(f'F1 Score: {f1_score:.2%}')

    # Save metrics to a text file
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "metrics.txt")
    with open(output_file_path, "w") as output_file:
        output_file.write("Confusion Matrix:\n")
        output_file.write(str(cm) + "\n\n")
        output_file.write(f'Precision: {precision:.2%}\n')
        output_file.write(f'Recall: {recall:.2%}\n')
        output_file.write(f'F1 Score: {f1_score:.2%}\n')

# Make Predictions
def predict_sentiment(text, classifier):
    words = word_tokenize(text)
    features = extract_features(words)
    sentiment = classifier.classify(features)
    return sentiment

def make_predictions(examples, classifier):
    predictions = []
    for example in examples:
        prediction = predict_sentiment(example, classifier)
        predictions.append(prediction)
        print(f'Example Prediction: {prediction}')

    return predictions

# Visualize Sentiment Distribution
def visualize_sentiment_distribution(predictions):
    sentiment_counts = {'pos': 0, 'neg': 0}
    for prediction in predictions:
        sentiment_counts[prediction] += 1

    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())

    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Predicted Sentiment Distribution for Examples')

    # Save the plot to an image file
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, "sentiment_distribution.png")
    plt.savefig(output_image_path)

    plt.show()
# Save Classifier Model
def save_model(classifier, filename="classifier_model.pickle"):
    with open(filename, 'wb') as model_file:
        pickle.dump(classifier, model_file)

# Load Classifier Model
def load_model(filename="classifier_model.pickle"):
    with open(filename, 'rb') as model_file:
        classifier = pickle.load(model_file)
    return classifier



if __name__ == "__main__":
    # Load Combined Movie Reviews Dataset
    reviews = load_movie_reviews()

    # Preprocess Data
    processed_reviews = preprocess_data(reviews)

    # Split Data
    train_data, test_data = split_data(processed_reviews)

    # Train Naive Bayes Classifier
    classifier = train_classifier(train_data)

    # Save Trained Model
    save_model(classifier)

    # Evaluate Model
    evaluate_model(classifier, test_data)

    # Calculate Metrics and Display Confusion Matrix
    calculate_metrics(classifier, test_data)
    loaded_classifier = load_model("classifier_model.pickle")

    # Make Predictions
    examples = [
        "This movie is fantastic! I loved every moment of it.",
        "The cinematography and acting were outstanding. A must-watch!",
        "The plot was confusing, and the characters were poorly developed.",
        "I regret watching this movie. It was a waste of time.",
        "I regret watching this. It was a waste of money."
    ]
    predictions = make_predictions(examples, loaded_classifier)  # Store predictions in a variable

    # Visualize Sentiment Distribution
    visualize_sentiment_distribution(predictions)