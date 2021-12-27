import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from io import StringIO
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix


path = "." # Set the path of the directory 

#CREATE NEW DATA SAMPLE--------------------------------------------------------
#filename_read = os.path.join(path, "DataSet.csv") # Create path (specified above) to DataSet.csv
#df = pd.read_csv(filename_read).fillna("c") # reads DataSet.csv to df, replaces empty cells with "c" to highlight missing data
#df = df.reindex(np.random.permutation(df.index)) # randomises the data
#print(df)

#mask = np.random.rand(len(df)) < 0.02 # 2% of dataset sample used(limited by available processing power)
#trainDF = pd.DataFrame(df[mask]) #20% training data 
#validationDF = pd.DataFrame(df[~mask]) #80% validation data

#filename_write = os.path.join(path, "DataSet_Sample.csv") #create path for csv file containing sample data
#trainDF.to_csv(filename_write, index=True) #write file

#READ SAMPLE FILE--------------------------------------------------------------

filename_read = os.path.join(path, "DataSet_Sample.csv")  # Create path (specified above) to DataSet.csv
data = pd.read_csv(filename_read) # Read csv file to data variable
data = data.reindex(np.random.permutation(data.index))    # Radomise entries

# Assign titles for columns in DataSet.csv
col = ['opinion', 'commentary']
data = data[col]
data.columns = ['opinion' , 'commentary']
data['negative'] = data['opinion'].factorize()[0] # Assign 0 to negative and 1 to positive

countv = CountVectorizer(ngram_range=(1, 2), min_df = 1) # CountVectorizer converts text to word count vectors looking at unigrams (1, 1) and bigrams (1, 2) 
features = countv.fit_transform(data.commentary).toarray()
x = features # Transformed commentary column assigned to x
y = data.negative # 0 or 1 assigned to y for each comment depending on opinion

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 ,random_state = 0) #universal settings for training-test split (20/80)    
                                    
#BASELINE----------------------------------------------------------------------
models = [LinearSVC(), KNeighborsClassifier(), RandomForestClassifier() ] # Baseline accuracy (training and test set) for each model
for model in models:
    model.fit(X_train, y_train)
    test_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    print('Baseline Training Accuracy: %.3f' % accuracy_score(y_train,test_pred )) # output baseline training accuracy
    print('Baseline Testing Accuracy: %.3f' % accuracy_score(y_test,y_pred)) # output baseline testing accuracy
    print('\n' )

#------------------------------------------------------------------------------

#METRICS CLASSIFICATION METHOD
def metrics_classification(y_test, y_pred, target_names = data['opinion'].unique()):
    print(metrics.classification_report(y_test, y_pred, target_names = target_names) )


#RUN MODEL---------------------------------------------------------------------

# Uncomment desired model
model = LinearSVC(C = 1, random_state = 0)
# model = RandomForestClassifier(n_estimators = 50 , criterion = "entropy")
# model = KNeighborsClassifier(n_neighbors = 1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Current Model Training Accuracy: %.2f' % model.score(X_train,y_train)) #output current training accuracy
print('Current Model Testing Accuracy: %.2f' % accuracy_score(y_test,y_pred)) #output current testing accuracy

kf = KFold(n_splits = 5, shuffle = True) # kFold Split (k=5)

trainingscores = []
testingscores = []
for i in range(5):
    result = next(kf.split(data), None)
    x_train = x[result[0]]
    x_test = x[result[1]]
    y_train = y.iloc[result[0]]
    y_test = y.iloc[result[1]]
    model = model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    testingscores.append(model.score(x_test,y_test))
    trainingscores.append(model.score(x_train,y_train))
    metrics_classification(y_test,predictions)
    cm = confusion_matrix ( y_test, predictions)
    print(cm) # output confusion matrix for each fold

#------------------------------------------------------------------------------

print('Training Scores: ', trainingscores) # output training scores
print('Testing Scores: ', testingscores) # output testing scores
print('Average Training Score:', np.mean(trainingscores)) # output average (mean) of training scores
print('Average K-Fold Score:', np.mean(testingscores)) # output average (mean) score across all folds