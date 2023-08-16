import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data from csv file

dt = r"C:\Users\LENOVO\Downloads\Compressed\FNews\mail_data.csv"
data = pd.read_csv(dt)

## print number of samples in the dataset 
print(len(data))

##print first lines of the dataset
print(data.head())

## print name of columns and thier types
print(data.dtypes)

## print dataset classes (target column values)
print(data.Category.unique())

## check if the dataset is balanced
print(data.Category.value_counts())


## check is the dataset contains missing values (NaN, valeurs manquantes)
print(data.isna().sum())

# Split data into feature and target
X = data['Message']
y = data['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Convert sparse matrix to dense numpy array
X_train_counts = X_train_counts.toarray()
X_test_counts = X_test_counts.toarray()

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Convert sparse matrix to dense numpy array
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# Vectorize the text data using existence-based approach
binary_vectorizer = CountVectorizer(binary=True)
X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

# Convert sparse matrix to dense numpy array
X_train_binary = X_train_binary.toarray()
X_test_binary = X_test_binary.toarray()

# Define the models
models = {
    'MultinomialNB-counts': MultinomialNB(),
    'MultinomialNB-tfidf': MultinomialNB(),
    'MultinomialNB-binary': MultinomialNB(),
    'GaussianNB-counts': GaussianNB(),
    'GaussianNB-tfidf': GaussianNB(),
    'GaussianNB-binary': GaussianNB(),
    'BernoulliNB-counts': BernoulliNB(),
    'BernoulliNB-tfidf': BernoulliNB(),
    'BernoulliNB-binary': BernoulliNB(),
}

# Train and test the models
for model_name, model in models.items():
    # Select the appropriate feature vector
    if 'counts' in model_name:
        X_train_feature = X_train_counts
        X_test_feature = X_test_counts
    elif 'tfidf' in model_name:
        X_train_feature = X_train_tfidf
        X_test_feature = X_test_tfidf
    elif 'binary' in model_name:
        X_train_feature = X_train_binary
        X_test_feature = X_test_binary
    else:
        raise ValueError('Invalid model name')

    # Select the appropriate Naive Bayes classifier
    if 'MultinomialNB' in model_name:
        clf = MultinomialNB()
    elif 'GaussianNB' in model_name:
        clf = GaussianNB()
    elif 'BernoulliNB' in model_name:
        clf = BernoulliNB()
    else:
        raise ValueError('Invalid model name')

    # Train the model
    clf.fit(X_train_feature, y_train)

    # Test the model
    y_pred = clf.predict(X_test_feature)

    # Calculate the accuracy score
    print("Model_NB: ", model_name)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
