import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# User profile indicating their interests
user_interests = [
    "sci.space",  # User's interest in the "sci.space" category
]

# Calculate cosine similarity between user interests and news articles
user_vector = tfidf_vectorizer.transform(user_interests)
similarities = cosine_similarity(user_vector, X_test_tfidf)

# Sort news articles by similarity score
article_scores = list(enumerate(similarities[0]))
article_scores.sort(key=lambda x: x[1], reverse=True)

# Number of recommended articles
num_recommendations = 3

# Display personalized news recommendations
print("\nPersonalized News Recommendations based on User's Interest:")
for i in range(num_recommendations):
    article_index, similarity_score = article_scores[i]
    recommended_article = X_test[article_index]
    print(f"Recommended Article {i + 1}: {recommended_article}")

# Plotting distribution of news categories using a pie chart
category_counts = np.bincount(y_test)
labels = [newsgroups.target_names[i] for i in range(len(category_counts))]

plt.figure(figsize=(10, 10))
plt.pie(category_counts, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Classification of various News Articles')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()