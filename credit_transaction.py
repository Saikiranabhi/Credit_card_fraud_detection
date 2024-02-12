#import the libraries required
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
#Defining the the relationship or mapping between data fields in a source dataset and their corresponding data fields in a target dataset or database schema
DATA_SOURCE_MAPPING = 'fraud-detection:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F817870%2F1399887%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240211%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240211T162344Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D32c781dde0e137d956989c4b164aecaa5d10e02ca17e0a08143d8b133c551ffec74d35f08835ea046f73353d25c82f22fa50976367d878c1506dbbe994e1711fe8b2949bc8202a061ceb490f8782161eefdf76fb14ca06856336b8ad72ef0da99b54a5526090be2bc55a8f80f1d7463d38ca12989deb34d3889a817d15c1f909ac9f06542f82c9379652f8d4d5062b5ab1d2c44b8fa0cfdd44a82a326e7465e54e036e5cea187d2b63b0a7bba901c7aac276bcec6dc88a7c04512ba9bf7650e0c75cb648fd64148d77ceedb6a421af700575615f32a2c676d6d57b986a1fc047988f3bc6ab180cb46045ac247188ea7b1dbbf3da884b9f32ffa2bdfab6efc129'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#read the data from the csv file

data = pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv", encoding='ISO-8859-1')

data

data.head()

data.isnull().sum()

# Example imputation for numeric columns
data['city_pop'].fillna(data['city_pop'].median(), inplace=True)
data['unix_time'].fillna(data['unix_time'].median(), inplace=True)
data['merch_lat'].fillna(data['merch_lat'].median(), inplace=True)
data['merch_long'].fillna(data['merch_long'].median(), inplace=True)
data['is_fraud'].fillna(0, inplace=True)  # Assuming is_fraud is a binary variable

data.dropna(subset=['unix_time', 'merch_lat', 'merch_long', 'is_fraud'], inplace=True)

# Check for missing values in the entire dataset
missing_values = data.isnull().sum()
print(missing_values)

#describe the data that is preprocessed

data.describe()

X = data.drop('is_fraud', axis=1)  # Assuming 'is_fraud' is the target variable
y = data['is_fraud']

data['Unnamed: 0'],unnamed_name=pd.factorize(data['Unnamed: 0'])
print(unnamed_name)

data['cc_num'],cc_name=pd.factorize(data['cc_num'])
print(cc_name)

data['category'],category_name=pd.factorize(data['category'])
print(category_name)

data['trans_date_trans_time'],time_name=pd.factorize(data['trans_date_trans_time'])
print(time_name)

data['amt'],amt_name=pd.factorize(data['amt'])
print(amt_name)

data['merchant'],merchant_name=pd.factorize(data['merchant'])
print(merchant_name)

data['zip'],zip_name=pd.factorize(data['zip'])
print(zip_name)

data['lat'],lat_name=pd.factorize(data['lat'])
print(lat_name)

data['long'],long_name=pd.factorize(data['long'])
print(long_name)

data['city_pop'],city_name=pd.factorize(data['city_pop'])
print(city_name)

data['unix_time'],unix_name=pd.factorize(data['unix_time'])
print(unix_name)

data['merch_lat'],merch_name=pd.factorize(data['merch_lat'])
print(merch_name)

data['merch_long'],long_name=pd.factorize(data['merch_long'])
print(long_name)

data['is_fraud'],fraud_name=pd.factorize(data['is_fraud'])
print(fraud_name)

data['first'],first_name=pd.factorize(data['first'])
print(first_name)

data['last'],last_name=pd.factorize(data['last'])
print(last_name)

data['street'],street_name=pd.factorize(data['street'])
print(street_name)

data['job'],job_name=pd.factorize(data['job'])
print(job_name)

data['dob'],dob_name=pd.factorize(data['dob'])
print(dob_name)

data['trans_num'],trans_name=pd.factorize(data['trans_num'])
print(trans_name)

data['gender'],gender_name=pd.factorize(data['gender'])
print(gender_name)

data['city'],city_name=pd.factorize(data['city'])
print(city_name)

data['state'],state_name=pd.factorize(data['state'])
print(state_name)

x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

#Train the model with the data that is already preprocessed

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

feature_names = x.columns

from sklearn import tree, metrics
dtree=tree.DecisionTreeClassifier(criterion='gini')#entrophy or gini
dtree.fit(x_train,y_train)

y_pred = dtree.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

classification_rep = classification_report(y_test, y_pred)

print("Decision Tree Model:")
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_rep)
