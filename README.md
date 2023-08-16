# NaiveBayes_With_Vectorizers
The code performs text classification using Naive Bayes models (MultinomialNB, GaussianNB, and BernoulliNB.) on a dataset. It preprocesses data, uses 3 vectorization techniques  (CountVectorizer, TF-IDF Vectorizer, and Binary Vectorizer. ), and evaluates 9 models. Each model's accuracy, classification report, and confusion matrix are printed. Valuable for learning text classification and experimenting with models.


Nine different Naive Bayes models (MultinomialNB, GaussianNB, and BernoulliNB) are defined, each using one of the three vectorization methods. The code iterates through these models, trains them on the training data, and evaluates their performance on the testing data. For each model, it prints the model name, accuracy score, classification report (including precision, recall, and F1-score), and confusion matrix.

Overall, this code demonstrates a comprehensive approach to text classification using various Naive Bayes models and vectorization techniques. It's a valuable resource for understanding and experimenting with different strategies for text classification tasks.



Resaults(outputs): 
              precision    recall  f1-score   support

         ham       0.99      1.00      1.00       966
        spam       1.00      0.94      0.97       149

    accuracy                           0.99      1115
   macro avg       1.00      0.97      0.98      1115
weighted avg       0.99      0.99      0.99      1115

Confusion Matrix:
[[966   0]
 [  9 140]]
Model_NB:  MultinomialNB-tfidf
Accuracy: 0.9650224215246637
Classification Report:
              precision    recall  f1-score   support

         ham       0.96      1.00      0.98       966
        spam       1.00      0.74      0.85       149

    accuracy                           0.97      1115
   macro avg       0.98      0.87      0.91      1115
weighted avg       0.97      0.97      0.96      1115

Confusion Matrix:
[[966   0]
 [ 39 110]]
Model_NB:  MultinomialNB-binary
Accuracy: 0.9910313901345291
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      1.00      0.99       966
        spam       1.00      0.93      0.97       149

    accuracy                           0.99      1115
   macro avg       0.99      0.97      0.98      1115
weighted avg       0.99      0.99      0.99      1115

Confusion Matrix:
[[966   0]
 [ 10 139]]
Model_NB:  GaussianNB-counts
Accuracy: 0.9067264573991032
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.90      0.94       966
        spam       0.60      0.93      0.73       149

    accuracy                           0.91      1115
   macro avg       0.79      0.91      0.84      1115
weighted avg       0.94      0.91      0.91      1115

Confusion Matrix:
[[873  93]
 [ 11 138]]
Model_NB:  GaussianNB-tfidf
Accuracy: 0.9049327354260089
Classification Report:
              precision    recall  f1-score   support

         ham       0.98      0.90      0.94       966
        spam       0.59      0.91      0.72       149

    accuracy                           0.90      1115
   macro avg       0.79      0.91      0.83      1115
weighted avg       0.93      0.90      0.91      1115

Confusion Matrix:
[[874  92]
 [ 14 135]]
Model_NB:  GaussianNB-binary
Accuracy: 0.9067264573991032
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.90      0.94       966
        spam       0.60      0.93      0.73       149

    accuracy                           0.91      1115
   macro avg       0.79      0.91      0.84      1115
weighted avg       0.94      0.91      0.91      1115

Confusion Matrix:
[[873  93]
 [ 11 138]]
Model_NB:  BernoulliNB-counts
Accuracy: 0.9802690582959641
Classification Report:
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       1.00      0.85      0.92       149

    accuracy                           0.98      1115
   macro avg       0.99      0.93      0.95      1115
weighted avg       0.98      0.98      0.98      1115

Confusion Matrix:
[[966   0]
 [ 22 127]]
Model_NB:  BernoulliNB-tfidf
Accuracy: 0.9802690582959641
Classification Report:
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       1.00      0.85      0.92       149

    accuracy                           0.98      1115
   macro avg       0.99      0.93      0.95      1115
weighted avg       0.98      0.98      0.98      1115

Confusion Matrix:
[[966   0]
 [ 22 127]]
Model_NB:  BernoulliNB-binary
Accuracy: 0.9802690582959641
Classification Report:
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       1.00      0.85      0.92       149

    accuracy                           0.98      1115
   macro avg       0.99      0.93      0.95      1115
weighted avg       0.98      0.98      0.98      1115

Confusion Matrix:
[[966   0]
 [ 22 127]]

5312
   age  height_cm  weight_kg  ...  sit-ups counts  broad jump_cm  Blass
0   27      172.3      75.24  ...              60            217      B
1   25      165.0      55.80  ...              53            229      A
2   31      179.6      78.00  ...              49            181      B
3   32      174.5      71.10  ...              53            219      B
4   28      173.8      67.70  ...              45            217      B

[5 rows x 11 columns]
age                          int64
height_cm                  float64
weight_kg                  float64
body fat_%                 float64
diastolic                  float64
systolic                     int64
gripForce                  float64
sit and bend forward_cm    float64
sit-ups counts               int64
broad jump_cm                int64
Blass                       object
dtype: object
['B' 'A']
B    2699
A    2613
Name: Blass, dtype: int64
age                        0
height_cm                  0
weight_kg                  0
body fat_%                 0
diastolic                  0
systolic                   0
gripForce                  0
sit and bend forward_cm    0
sit-ups counts             0
broad jump_cm              0
Blass                      0
dtype: int64
Best parameters:  {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
Accuracy score:  0.7431796801505174
Classification report: 
               precision    recall  f1-score   support

           A       0.78      0.65      0.71       512
           B       0.72      0.83      0.77       551

    accuracy                           0.74      1063
   macro avg       0.75      0.74      0.74      1063
weighted avg       0.75      0.74      0.74      1063

Confusion matrix: 
 [[333 179]
 [ 94 457]]


