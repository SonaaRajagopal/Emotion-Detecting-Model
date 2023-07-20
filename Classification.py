#Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#Load and preprocess the data
def cleaner(row):
    if row in [np.nan, 'nan', 0.0]:
        return row
    else:
        x, y = 6, -6
        try:
            return float(row[x:y])
        except:
            print(f'{row, type(row)}, exception')

def pre_process(df, mode):
    df = df.drop(['UID', 'ph_no', 'cvv', 'credit_card_number', 'job', 'email', 'url', 'country', 'emoji', 'name'], axis=1)
    n = len(df.columns) if mode == 'test' else -1
    for i in df.columns[:n]:
        df[i] = df[i].apply(cleaner)
        print(f'{i} - Success!!!')
    df = df.fillna(df.median(numeric_only=True))
    return df

# Load and preprocess the training dataset
df = pd.read_csv('train_dataset.csv')
df = pre_process(df, 'train')

# Encode the target variable 'state'
le = LabelEncoder()
df['state'] = le.fit_transform(df['state'])

# Split the data into training and testing sets
x = df.drop('state', axis=1)
y = df['state']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create a Random Forest Classifier and train it on the training data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Evaluate the model's performance on the test data
accuracy = rf_model.score(x_test, y_test)
print(f"Random Forest Accuracy: {accuracy}")

# Load and preprocess the test dataset
df_test = pd.read_csv('test.csv')
df_test_cleaned = pre_process(df_test, 'test')

# Make predictions on the test dataset using the trained Random Forest model
y_pred = rf_model.predict(df_test_cleaned)

# Inverse transform the predicted labels to their original form
y_pred = le.inverse_transform(y_pred)

# Create the submission DataFrame
submission = pd.DataFrame({"UID": df_test['UID'], "state": y_pred})
# Save the submission to a CSV file
submission.to_csv('submission4.csv', index=False)
