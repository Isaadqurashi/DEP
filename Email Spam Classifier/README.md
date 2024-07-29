**TASK2**
#
# Spam SMS Classifier

This project is a machine learning application for classifying SMS messages as spam or non-spam using algorithms such as Naive Bayes. The project involves data preprocessing, model training, and prediction on new messages.

## Table of Contents
- [Installation](#installation)
- [Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](https://github.com/Isaadqurashi/DEP/blob/main/Email%20Spam%20Classifier/LICENSE)
- [Appendices](#appendices)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Isaadqurashi/DEP/tree/main/Email%20Spam%20Classifier
    cd spam-sms-classifier
    ```

2. Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn nltk scikit-learn
    ```

3. Download NLTK stopwords:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Dataset

The dataset used for this project is `Spam SMS Collection.csv`, which contains SMS messages labeled as spam or ham (non-spam).

Download 
[Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)


## Data Preprocessing

1. Load the dataset:
    ```python
    sms = pd.read_csv('Spam SMS Collection.csv', names=['label','message'])
    ```

2. Remove duplicates and reset the index:
    ```python
    sms.drop_duplicates(inplace=True)
    sms.reset_index(drop=True, inplace=True)
    ```

3. Visualize the distribution of labels:
    ```python
    plt.figure(figsize=(8,5))
    sns.countplot(x='label', data=sms)
    plt.xlabel('SMS Classification')
    plt.ylabel('Count')
    plt.show()
    ```

4. Clean and preprocess the messages:
    - Remove special characters
    - Convert to lowercase
    - Remove stopwords
    - Perform stemming

## Model Training

1. Create a Bag of Words model using `CountVectorizer`:
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(corpus).toarray()
    ```

2. Encode the labels:
    ```python
    y = pd.get_dummies(sms['label'])
    y = y.iloc[:, 1].values
    ```

3. Split the data into training and test sets:
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    ```

## Evaluation

1. Train and evaluate a Naive Bayes classifier with different alpha values:
    ```python
    best_accuracy = 0.0
    alpha_val = 0.0
    for i in np.arange(0.0, 1.1, 0.1):
        temp_classifier = MultinomialNB(alpha=i)
        temp_classifier.fit(X_train, y_train)
        temp_y_pred = temp_classifier.predict(X_test)
        score = accuracy_score(y_test, temp_y_pred)
        if score > best_accuracy:
            best_accuracy = score
            alpha_val = i
    ```

2. Fit the final model with the best alpha value:
    ```python
    classifier = MultinomialNB(alpha=alpha_val)
    classifier.fit(X_train, y_train)
    ```

3. Evaluate the accuracy of the model:
    ```python
    y_pred = classifier.predict(X_test)
    acc_s = accuracy_score(y_test, y_pred) * 100
    print("Accuracy Score {} %".format(round(acc_s, 2)))
    ```

## Prediction

1. Define a function to predict if a message is spam or not:
    ```python
    def predict_spam(sample_message):
        sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_message)
        sample_message = sample_message.lower()
        sample_message_words = sample_message.split()
        sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
        ps = PorterStemmer()
        final_message = [ps.stem(word) for word in sample_message_words]
        final_message = ' '.join(final_message)
        temp = cv.transform([final_message]).toarray()
        return classifier.predict(temp)
    ```

2. Test the function with sample messages:
    ```python
    result = ['Wait a minute, this is a SPAM!', 'Ohhh, this is a normal message.']
    msg = "Hi! You are pre-qualified for Premium SBI Credit Card. Also get Rs.500 worth Amazon Gift Card*, 10X Rewards Point* & more. Click"
    if predict_spam(msg):
        print(result[0])
    else:
        print(result[1])
    ```

## Usage

Run the Jupyter Notebook `Spam SMS Detection.ipynb` to see the entire workflow and test the classifier on new messages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](ttps://github.com/Isaadqurashi/DEP/blob/main/Email%20Spam%20Classifier/LICENSE) for more details.

## Appendices

- **Dataset:** `Spam SMS Collection.csv`
- **Code:** `Spam SMS Detection.ipynb`
