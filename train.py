import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset to train the AI
data = {
    'review': [
        'Excellent product, very happy', 'Highly recommended', 'Amazing quality',
        'Worst experience ever', 'Fake product, do not buy', 'Horrible quality',
        'Waste of money', 'Best purchase this year', 'Authentic item'
    ],
    'label': [1, 1, 1, 0, 0, 0, 0, 1, 1]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Create the Vectorizer (converts text to numbers)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])

# Create and Train the Model
model = MultinomialNB()
model.fit(X, df['label'])

# Save both files as .pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Success! model.pkl and vectorizer.pkl have been created.")