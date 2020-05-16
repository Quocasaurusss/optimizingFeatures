import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random

'''output of modified():
['OverallQual', 'TotalBsmtSF', 'GrLivArea']
Largest mae: 64733.003671908074
Smallest mae: 22730.87225968933
'''
'''output of create_trees when given about 11000 iterations
['MiscVal', 'GarageCars', 'TotalBsmtSF', 'OverallQual', 'GrLivArea']
Largest mae: 62710.8363983624
Smallest mae: 21875.897325467282
'''
#this line of code removes missing data from the data set
#it also removes data that isn't int64 because it such
#data types were making it difficult to train the model.

def clean_data():
    file_path = 'C:/Users/Quocamole/Documents/Machine Learning in Python/Choosing 5 Best Features/train.csv'
    home_data = pd.read_csv(file_path)
    #drops columns that have any null values
    home_data = home_data.dropna(axis='columns', how='any')
    remove_columns = []
    for i in home_data.columns:
         if home_data[i].dtypes != np.int64(1):
             remove_columns.append(i)
    home_data = home_data.drop(columns=remove_columns)
    return home_data

def possible_features_combos():
    #stores a version of data set with missing data removed
    #this version of data set will also only contain features that are of type int64
    home_data = clean_data()
    #stores combinations of features where order matters, but repitition not allowed.
    possible_combos = []
    home_data = home_data.drop(columns=['SalePrice'])
    #this while loop makes sure to access all possible permutations
    iteration = 0
    
    #possible number of combinations is equal to 278256
    #278256/4
    while len(possible_combos) != 1000:
        iteration += 1
        same = 0
        features = [1, 2, 3, 4, 5]
        count = len(home_data.columns) ** len(features)
        indices = random.sample(range(len(home_data.columns)), len(features))
        for i in range(len(features)):
            features[i] = home_data.columns[indices[i]]
        #this checks to see if the created combination of features already exists in our 
        #array of all possible combinations.
        for combo in possible_combos:
            for feature in combo:
                if feature in features:
                    same += 1
                if same == len(features):
                    #print('it already exists')
                    count -= 1
                    break
            same = 0
            if count != len(home_data.columns) ** len(features):
                #print('iteration', iteration, 'just broke')
                break

        if count == len(home_data.columns) ** len(features):
            #print('iteration', iteration, 'just added')
            possible_combos.append(features)
        print('SIZE', len(possible_combos))
    print('number of iterations', iteration)
    return possible_combos

def create_trees():
    home_data = clean_data()
    possible = possible_features_combos()
    smallest = 1000000000000 #current smallest MAE value
    largest = 0 #current largest MAE value
    bestfeatures = [] #the features that generated our current smallest MAE value
    count = 0
    for i in possible:
        #found 22769 at iteration 8569
        count += 1
        print(count, smallest)
        y = home_data.SalePrice #the data we will predict
        X = home_data[i] #the features we will use to train model
        #splitting data into training and validation data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
        ##creating our Decision tree regressor model
        model = DecisionTreeRegressor(max_leaf_nodes = 25, random_state=1)
        #using the training data to train our model
        model.fit(train_X, train_y)
        #using our model to predict the Sale Price
        prediction = model.predict(val_X)
        #computing the mean absolute error by comparing 
        #the sale price we predicted vs the actual sale price
        mae = mean_absolute_error(prediction, val_y)
        if mae < smallest: #updates smallest mae value if another smaller one exists
            smallest = mae
            bestfeatures = i
        if mae > largest: #updates largest mae value if another larger one exists
            largest = mae
            
    print(bestfeatures)
    print('Largest mae:', largest)
    print('Smallest mae:',smallest)
    
def modified():
    '''variables'''
    #stores a version of data set with missing data removed
    #this version of data set will also only contain features that are of type int64
    home_data = clean_data()
    #the value that we will predict is SalePrice of dataset
    y = home_data.SalePrice
    #when we select features as independent variables, we want to make sure
    #that the value we are trying to predict (SalePrice) isn't one of them
    home_data = home_data.drop(columns=['SalePrice'])
    #39270 possible permutations
    #6545 possible combinations, no repition, order does not matter (binomial coefficient)]
    smallest = 1000000000000 #current smallest MAE value
    largest = 0 #current largest MAE value
    bestfeatures = [] #the features that generated our current smallest MAE value
    #this performs a brute force search on every possible permutation 
    iteration = 0
    #it found the smallest value in 3824 iterations
    
    #this series of for loops generates combinations of features
    #where order matters and repetition is allowed
    for a in home_data.columns:
        for b in home_data.columns:
            for c in home_data.columns:
                for d in home_data.columns:
                    for e in home_data.columns:
                        iteration += 1
                        print(iteration, smallest)    
                        features = [a, b, c, d, e]
                        X = home_data[features]
                        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
                        model = DecisionTreeRegressor(max_leaf_nodes = 25, random_state=1)
                        model.fit(train_X, train_y)
                        prediction = model.predict(val_X)
                        mae = mean_absolute_error(prediction, val_y)
                        if mae < smallest:
                            smallest = mae
                            bestfeatures = features
                        if mae > largest:
                            largest = mae
            
    print(bestfeatures)
    print('Largest mae:', largest)
    print('Smallest mae:',smallest)

'''
Insights: 
using all 35 features is not optimal. It generates an mae of 24052
'''

'''METHOD CALLS'''
#possible_features_combos()
#create_trees()
modified() #found the smallest value at iteration 
#both() #31 seconds for input of 2000
#create_trees() #30 seconds for input of 2000
                
            

