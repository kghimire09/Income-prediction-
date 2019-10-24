def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

income_data=pd.read_csv('income.csv', header=0, delimiter=', ')
print(income_data.iloc[0])

labels=income_data[["income"]]
income_data['sex-int']=income_data['sex'].apply(lambda x: 0 if x == 'Male' else 1)
income_data['country-int']=income_data['native-country'].apply(lambda y: 0 if y == 'United-State' else 1)
data = income_data[['age','capital-gain','capital-loss','hours-per-week','sex-int','country-int']]

train_data,test_data,train_labels,test_labels=train_test_split(data, labels, random_state=1)

forest=RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
#print(forest.feature_importances_)
print(forest.score(test_data, test_labels))

#comparison with Decision Tree
tree=DecisionTreeClassifier(random_state=1)
tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels))



#print(income_data['native-country'].value_counts())




