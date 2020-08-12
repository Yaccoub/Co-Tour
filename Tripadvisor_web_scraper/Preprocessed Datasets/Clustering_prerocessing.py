import pandas as pd


def preprocessing(dataframe):
    df = dataframe.copy()
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Trip type: ': ''}, regex=True)
    df['date']= [datetime.strptime(date, '%B %Y')for date in df['date']]
    df = df.sort_values(by='date', ascending=False, inplace=False, ignore_index=True)
    df = df.set_index('date')

    return df

def clustering_process(dataframe):
    df = dataframe.copy()
    df[['city', 'country', 'extra']] = df['visitor_origin'].str.split(',', expand=True, n=2)
    #df = pd.get_dummies(df, columns=["visit", "country"])
    df = df.drop(['rating','title','text'], axis=1)
    return df

def feature_extraction(dataframe):
    df = dataframe.copy()
    df = preprocessing(df)
    df = clustering_process(df)
    visitors_by_country = df.groupby('country').count().sort_values('visit', ascending=True)['visitor_origin']
    type_of_visitors    = df.groupby('visit').count().sort_values('country', ascending=True)['visitor_origin']
    visitors_by_city    = df.groupby('city').count().sort_values('visit', ascending=True)['visitor_origin']