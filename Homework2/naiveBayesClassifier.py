import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# function to pull files in train and test sets and organize them for CountVectorizer() 
def loadDocumentsFromDirectory(directoryPath):
    documents = []
    labels = []
    category_counts = {'rec.autos': 0, 'comp.graphics': 0}
    
    for category in ['rec.autos', 'comp.graphics']:
        categoryPath = directoryPath + "/" + category
        for filename in os.listdir(categoryPath):
            with open(categoryPath + "/" + filename, 'r', errors='replace') as file:
                documents.append(file.read())
                labels.append(category)
                category_counts[category] += 1
                
    return documents, labels, category_counts

# run train and test
trainDocuments, trainLabels, trainCategoryCounts = loadDocumentsFromDirectory('20news-bydate/20news-bydate-train')
testDocuments, testLabels, testCategoryCounts = loadDocumentsFromDirectory('20news-bydate/20news-bydate-train')

# Convert documents to token count matrix, excluding numerical tokens and underscores
vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W_]+\b')
XTrain = vectorizer.fit_transform(trainDocuments)
XTest = vectorizer.transform(testDocuments)

# Print vocabulary size and document counts
print(f"\nSize of the vocabulary: {len(vectorizer.get_feature_names_out())}")
print(f"Training set documents per category: {trainCategoryCounts}")
print(f"Test set documents per category: {testCategoryCounts}")

# setup classifier and test
classifier = MultinomialNB()
classifier.fit(XTrain, trainLabels)
testPredictions = classifier.predict(XTest)

# Calculate F1-Scores
f1ScoreRecAutos = f1_score(testLabels, testPredictions, pos_label='rec.autos')
f1ScoreCompGraphics = f1_score(testLabels, testPredictions, pos_label='comp.graphics')
print(f"\nF1-score (rec.autos as positive): {f1ScoreRecAutos}")
print(f"F1-score (comp.graphics as positive): {f1ScoreCompGraphics}")

# Calculate and print accuracy on the training set
trainAccuracy = classifier.score(XTrain, trainLabels)
print(f"Training set accuracy: {trainAccuracy * 100:.2f}%")

# Calculate and print accuracy on the test set
testAccuracy = classifier.score(XTest, testLabels)
print(f"Test set accuracy: {testAccuracy * 100:.2f}%")