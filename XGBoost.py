import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Define the cleaner function
def cleaner(row):
    if row in [np.nan, 'nan', 0.0]:
        return row
    else:
        x, y = 6, -6
        try:
            return float(row[x:y])
        except:
            print(f'{row, type(row)}, exception')

# Define the pre_process function
def pre_process(df, mode):
    df = df.drop(['UID', 'ph_no', 'cvv', 'credit_card_number', 'job', 'email', 'url', 'country', 'emoji', 'name'], axis=1)
    n = len(df.columns) if mode == 'test' else -1
    for i in df.columns[:n]:
        df[i] = df[i].apply(cleaner)
        print(f'{i} - Success!!!')
    df = df.fillna(df.median(numeric_only=True))
    return df

# Read the training dataset
df_train = pd.read_csv('/kaggle/input/testcsv/train_dataset.csv')

# Preprocess the training dataset
df_train = pre_process(df_train, 'train')

# Extract features and labels
x_train = df_train.drop('state', axis=1)
y_train = df_train['state']

# Label Encoding for the target variable 'state'
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

# Build the XGBoost Classifier model
xgb_model = xgb.XGBClassifier()

# Train the XGBoost model
xgb_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = xgb_model.predict(x_test)

# Calculate the accuracy of the XGBoost model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Read the test dataset
df_test = pd.read_csv('/kaggle/input/testcsv/test.csv')

# Preprocess the test dataset
df_test_cleaned = pre_process(df_test, 'test')

# Make predictions on the test data using the trained XGBoost model
y_pred = xgb_model.predict(df_test_cleaned)

# Inverse transform the encoded predictions to get the original 'state' labels
y_pred_labels = le.inverse_transform(y_pred)

# Create the submission DataFrame
submission = pd.DataFrame({"UID": df_test['UID'], "state": y_pred_labels})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission_xgb.csv', index=False)
