# ----------------------------------- Question 1 : -----------------------------------
# Training data
training_data = [
    ("it is raining", "rainy"),
    ("picnic on a hot afternoon", "sunny"),
    ("they wore sunglasses", "sunny"),
    ("going out with an umbrella", "rainy")
]

# Split expressions into words and compute word frequencies for each category
word_freq_sunny = {}
word_freq_rainy = {}
category_count = {"sunny": 0, "rainy": 0}

for expression, category in training_data:
    words = expression.split()
    category_count[category] += 1
    for word in words:
        if category == "sunny":
            word_freq_sunny[word] = word_freq_sunny.get(word, 0) + 1
        else:
            word_freq_rainy[word] = word_freq_rainy.get(word, 0) + 1

# Calculate prior probabilities
total_samples = len(training_data)
P_sunny = category_count["sunny"] / total_samples
P_rainy = category_count["rainy"] / total_samples

# Function to calculate conditional probability of a word given a category with Laplace smoothing
def conditional_probability(word, category):
    smoothing_factor = 1  # Laplace smoothing
    if category == "sunny":
        return (word_freq_sunny.get(word, 0) + smoothing_factor) / (category_count["sunny"] + smoothing_factor * len(word_freq_sunny))
    else:
        return (word_freq_rainy.get(word, 0) + smoothing_factor) / (category_count["rainy"] + smoothing_factor * len(word_freq_rainy))

# Expression to be classified
expression1 = "a cone of ice cream"
expression2 = "a cup of hot coffee"

# Calculate conditional probabilities for each word in the expressions
words1 = expression1.split()
words2 = expression2.split()

P_sunny_given_expression1 = P_sunny
P_rainy_given_expression1 = P_rainy

P_sunny_given_expression2 = P_sunny
P_rainy_given_expression2 = P_rainy

for word in words1:
    P_sunny_given_expression1 *= conditional_probability(word, "sunny")
    P_rainy_given_expression1 *= conditional_probability(word, "rainy")

for word in words2:
    P_sunny_given_expression2 *= conditional_probability(word, "sunny")
    P_rainy_given_expression2 *= conditional_probability(word, "rainy")

# Normalize the probabilities
total_probability_expression1 = P_sunny_given_expression1 + P_rainy_given_expression1
P_sunny_given_expression1 /= total_probability_expression1
P_rainy_given_expression1 /= total_probability_expression1

total_probability_expression2 = P_sunny_given_expression2 + P_rainy_given_expression2
P_sunny_given_expression2 /= total_probability_expression2
P_rainy_given_expression2 /= total_probability_expression2

print("P(sunny|a cone of ice cream) =", P_sunny_given_expression1)
print("P(rainy|a cup of hot coffee) =", P_rainy_given_expression2)



# ----------------------------------- Question 2 : -----------------------------------
import argparse
import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k):
        self.k = k
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(float))

    def train(self, data, labels):
        total_samples = len(data)
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_mask = labels == label
            self.class_probs[label] = (np.sum(label_mask) + self.k) / (total_samples + len(unique_labels) * self.k)
            
            for feature in range(data.shape[1]):
                feature_count = np.sum(data[label_mask][:, feature])
                total_feature_count = np.sum(data[:, feature])
                self.feature_probs[label][feature] = (feature_count + self.k) / (total_feature_count + 2 * self.k)

    def predict(self, data):
        predictions = []
        for sample in data:
            log_probs = defaultdict(float)
            for label, class_prob in self.class_probs.items():
                log_probs[label] = np.log(class_prob)
                for feature, feature_value in enumerate(sample):
                    log_probs[label] += np.log(self.feature_probs[label][feature]) if feature_value else np.log(1 - self.feature_probs[label][feature])
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-autotune', action='store_true')
    args = parser.parse_args()

    data, labels = load_data()  # Load your data here

    if args.autotune:
        best_k, best_accuracy = None, 0
        kgrid = [1, 2, 3, 4, 5]  # Adjust this to include your desired k values
        for k in kgrid:
            train_data, val_data, train_labels, val_labels = split_data(data, labels)
            classifier = NaiveBayesClassifier(k)
            classifier.train(train_data, train_labels)
            predictions = classifier.predict(val_data)
            accuracy = np.mean(predictions == val_labels)
            print(f'k={k}, Validation Accuracy: {accuracy * 100:.2f}%')
            if accuracy > best_accuracy:
                best_k, best_accuracy = k, accuracy
        print(f'Best k: {best_k}')

    else:
        train_data, test_data, train_labels, test_labels = split_data(data, labels)
        classifier = NaiveBayesClassifier(args.k)
        classifier.train(train_data, train_labels)
        predictions = classifier.predict(test_data)
        accuracy = np.mean(predictions == test_labels)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
