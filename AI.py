import numpy as np
import pandas as pd
from scipy import stats
import pickle
import warnings
# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Filter warnings
warnings.filterwarnings('ignore') #filter warnings
# Show plots inline
%matplotlib inline

# Load datasets
url_ping = 'https://drive.google.com/file/d/1r4dJxXOW7dSSspucCe51XRcn2IMYhiwa'
url_ping = 'https://drive.google.com/uc?id=' + url_ping.split('/')[-1]
ping_df = pd.read_csv(url_ping, delimiter='\t')

url_voice = 'https://drive.google.com/file/d/1idNUrXQEtiTkCiVtV3yHixX6XZivgtGF'
url_voice = 'https://drive.google.com/uc?id=' + url_voice.split('/')[-1]
voice_df = pd.read_csv(url_voice, delimiter='\t')

url_dns = 'https://drive.google.com/file/d/1mj7KYS7flCHZO7KxiYshgTXhe6j78XPC'
url_dns = 'https://drive.google.com/uc?id=' + url_dns.split('/')[-1]
dns_df = pd.read_csv(url_dns, delimiter='\t')

url_telnet = 'https://drive.google.com/file/d/11B8_7XEtDzjZk1VevUieVOrcLLLozv20'
url_telnet = 'https://drive.google.com/uc?id=' + url_telnet.split('/')[-1]
telnet_df = pd.read_csv(url_telnet, delimiter='\t')

df = pd.concat([ping_df, voice_df, dns_df, telnet_df], ignore_index=True)
df.dropna(inplace=True)
df.drop('Forward Packets', axis=1, inplace=True)
df.drop('Forward Bytes', axis=1, inplace=True)
df.drop('Reverse Packets', axis=1, inplace=True)
df.drop('Reverse Bytes', axis=1, inplace=True)
df['Traffic Type'] = df['Traffic Type'].astype('category')
df['Traffic Type'].cat.categories
df['Traffic Type'].cat.codes.head()


# Split dataset
X = df.drop('Traffic Type',axis=1)
y = df['Traffic Type']

print(df.columns)
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.9, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function to plot confusion matrix
def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    plt.xlabel('True Label', fontsize=15)
    plt.ylabel('Predicted Label', fontsize=15)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            color = 'black'
            if cm[i][j] > 5:
                color = 'white'
            plt.text(j, i, cm[i][j], horizontalalignment='center', color=color, fontsize=15)

# Define a function to plot roc curve
def plot_roc_curve(y_test, y_score, title, labels):
    n_classes = len(labels)
    y_test = label_binarize(y_test, classes=labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test[:, i], y_score[:, i])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title(title, fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

# Define a function to plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=15)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best", fontsize=12)
    return plt

# Create a list of models
models = [RandomForestClassifier(), GaussianNB(), LogisticRegression(), KNeighborsClassifier(), MLPClassifier()]

# Loop through the models and evaluate them
for model in models:
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
    # Print the accuracy score
    print('Accuracy %s : %.5f%%' % (model.__class__.__name__, accuracy_score(predictions, y_test)*100))
    # Print the classification report
    print(classification_report(predictions, y_test,digits=5))
    # Print the confusion matrix
    cm = confusion_matrix(predictions, y_test, labels=y.cat.categories)
    print(cm)
    # Plot the confusion matrix
    plot_confusion_matrix(cm, 'Confusion Matrix %s' % model.__class__.__name__, y.cat.categories)
    # Get the probability scores
    y_score = model.predict_proba(X_test)
    # Plot the roc curve
    plot_roc_curve(y_test, y_score, 'ROC Curve %s' % model.__class__.__name__, y.cat.categories)
    # Plot the learning curve
    plot_learning_curve(model, 'Learning Curve %s' % model.__class__.__name__, X, y, cv=5)
    plt.show()
    # Save the model
    pickle.dump(model, open(model.__class__.__name__, 'wb'))
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to get user input
def get_user_input(feature_names):
    user_data = []
    print("Enter the following information:")

    for feature in feature_names:
        value = input(f"{feature}: ")
        user_data.append(float(value))

    return np.array(user_data).reshape(1, -1)

# Function to preprocess user input
def preprocess_user_input(user_data, scaler):
    # Transform user input using the fitted scaler
    user_data_scaled = scaler.transform(user_data)
    return user_data_scaled

# New feature names
feature_names = ['Delta Forward Packets', 'Delta Forward Bytes',
       'Forward Instantaneous Packets per Second',
       'Forward Average Packets per second',
       'Forward Instantaneous Bytes per Second',
       'Forward Average Bytes per second', 'Delta Reverse Packets',
       'Delta Reverse Bytes', 'DeltaReverse Instantaneous Packets per Second',
       'Reverse Average Packets per second',
       'Reverse Instantaneous Bytes per Second',
       'Reverse Average Bytes per second']

# Get user input
user_input = get_user_input(feature_names)

# Create a new StandardScaler without providing feature names during fit
scaler_without_names = StandardScaler()
scaler_without_names.fit(np.zeros_like(user_input))

# Preprocess user input
user_input_scaled = preprocess_user_input(user_input, scaler_without_names)

print("\n")

# List of models
models = [RandomForestClassifier(), GaussianNB(), LogisticRegression(), KNeighborsClassifier(), MLPClassifier()]

# Predict using each model
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train) # Assuming X_train and y_train are defined
    model_prediction = model.predict(user_input_scaled)
    print(f"{model_name} Prediction: {model_prediction[0]}")
