import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
​
def cleaner(row):
    if row in [np.nan, 'nan', 0.0]:
        return row
    else:
        x, y = 6, -6
        try:
            return float(row[x:y])
        except:
            print(f'{row, type(row)}, exception')
​
def pre_process(df, mode):
    df = df.drop(['UID', 'ph_no', 'cvv', 'credit_card_number', 'job', 'email', 'url', 'country', 'emoji', 'name'], axis=1)
    n = len(df.columns) if mode == 'test' else -1
    for i in df.columns[:n]:
        df[i] = df[i].apply(cleaner)
        print(f'{i} - Success!!!')
    df = df.fillna(df.median(numeric_only=True))
    return df
​
# Load and preprocess the training dataset
df = pd.read_csv('train_dataset.csv')
df = pre_process(df, 'train')
​
# Perform clustering
x_cluster = df.drop('state', axis=1)
num_clusters = 3  # Choose the number of clusters as per your requirement
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(x_cluster)
cluster_labels = kmeans.labels_
df['cluster'] = cluster_labels
​
# Encode the target variable 'state'
le = LabelEncoder()
df['state'] = le.fit_transform(df['state'])
​
# Feature Scaling
scaler = StandardScaler()
df[df.columns.drop(['state', 'cluster'])] = scaler.fit_transform(df[df.columns.drop(['state', 'cluster'])])
​
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.drop(['state', 'cluster'], axis=1), df['state'], test_size=0.3, random_state=42)
​
# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
​
# Evaluate the model on the test set
y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
​
​
# Load and preprocess the test dataset
df_test = pd.read_csv('test.csv')
df_test_cleaned = pre_process(df_test, 'test')
​
# Feature Scaling for the test set
#df_test_cleaned[df_test_cleaned.columns.drop('cluster')] = scaler.transform(df_test_cleaned[df_test_cleaned.columns.drop('cluster')])
df_test_cleaned.head()
len(df_test_cleaned.columns)
df_test_cleaned.isna().sum().sum()
​
# Predict the clusters for the test set
test_cluster_labels = kmeans.predict(df_test_cleaned)
df_test_cleaned['cluster'] = test_cluster_labels
​
# Make predictions using the trained Random Forest model
y_pred = rf_model.predict(df_test_cleaned.drop('cluster', axis=1))
y_pred = le.inverse_transform(y_pred)
​
# Create the submission DataFrame
submission = pd.DataFrame({"UID": df_test['UID'], "state": y_pred})
​
# Save the submission to a CSV file
submission.to_csv('submission1.csv', index=False)
​
