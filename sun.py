import pandas as pd

import numpy as np

#df=pre_process(df,'train')

df=pd.read_csv('train_dataset.csv')

df.head()

df.info()

print(list(df.columns))

df.isna().sum()

df.isna().sum().sum()

def cleaner(row):
    
    if row in [np.nan,'nan',0.0]:
        return row
    else:
        x,y=6,-6
        try:
            return float(row[x:y])
        except:
            print(f'{row, type(row)}, exception')

def pre_process(df,mode):
    df=df.drop(['UID', 'ph_no', 'cvv', 'credit_card_number', 'job', 'email', 'url', 'country', 'emoji', 'name'],axis=1)
    n=len(df.columns) if mode=='test' else -1
    for i in df.columns[:n]:
        df[i]=df[i].apply(cleaner)
        print(f'{i} - Success!!!')
    df=df.fillna(df.median(numeric_only=True))
    return df

df=pre_process(df,'train')

df.select_dtypes(include=['object']).columns

df.isna().sum()

df.isna().sum().sum()

from sklearn.preprocessing import LabelEncoder as LE
x=df.drop('state',axis=1)
y=df['state']
le=LE()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
len(x_train),len(y_train),len(x_test),len(y_test)

x_train

from sklearn.linear_model import LogisticRegression as LR
lr=LR()
lr.fit(x_train,y_train)

lr.score(x_test,y_test)

df_test=pd.read_csv('test_dataset.csv')
df_test_cleaned=pre_process(df_test,'test')

df_test_cleaned.head()

len(df_test_cleaned.columns)

df_test_cleaned.isna().sum().sum()

y_pred=lr.predict(df_test_cleaned)

y_pred=le.inverse_transform(y_pred)

submission=pd.DataFrame({"UID":df_test['UID'],"state":y_pred})

submission.head()

submission.to_csv('submission.csv',index=False)