import pandas as pd
from io import StringIO
import sys

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:

if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

print(df.dropna(axis=0))

print(df.dropna(axis=1))

print(df.dropna(how='all'))

print(df.dropna(thresh=4))

print(df.dropna(subset=['C']))

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)

imputed_data = imr.transform(df.values)
print(imputed_data)

print(df.fillna(df.mean()))



# 4.2.2

import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
print(df['classlabel'].map(inv_class_mapping))


from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)
print(y)

X  = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
print(X)
print(X[:, 0])
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)


from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values

color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())


from sklearn.compose import ColumnTransformer

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]), ('nothing', 'passthrough', [1, 2])])
print(c_transf.fit_transform(X).astype(float))

print(pd.get_dummies(df[['price', 'color', 'size']]))

print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))