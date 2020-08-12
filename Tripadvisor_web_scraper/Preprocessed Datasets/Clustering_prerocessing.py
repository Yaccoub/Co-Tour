import pandas as pd


def preprocessing(dataframe):
    df = dataframe.copy()
    df['date'] = df['date'].replace({'Erlebnisdatum: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Reiseart: ': ''}, regex=True)
    df[['city', 'country']] = df['visitor_origin'].str.split(',', expand=True, n=2)
    df = pd.get_dummies(df, columns=["visit", "country"])
    return df

def clustering_preprocessing(dataframe):
    df = dataframe.copy()
    df['date'] = df['date'].replace({'Erlebnisdatum: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Reiseart: ': ''}, regex=True)
    df[['city', 'country']] = df['visitor_origin'].str.split(',', expand=True, n=2)
    df = pd.get_dummies(df, columns=["visit", "country"])
    df = df.drop(['rating'], axis=1)
    return df