import pandas as pd
import numpy as np
import glob
from pathlib import Path

path = "../Tripadvisor_web_scraper/data/*.csv"
dataframe = pd.DataFrame()
for fname in glob.glob(path):
    x = pd.read_csv(fname, low_memory=False)
    x = x.dropna(subset=['date'])
    x['date'] = [date.replace('Erlebnisdatum: ', '') for date in x['date']]
    x['place'] = Path(fname).stem
    x['visit'].fillna('', inplace=True)
    x['visit'] = [visit_type.replace('Reiseart: ', '') for visit_type in x['visit']]
    x = x[['date','place','rating', 'visit']]
    dataframe = pd.concat([dataframe, x], axis=0)

df = dataframe.groupby(['date','place', 'visit'] ,  as_index=False) [['rating']].mean()
df2 = dataframe.groupby(['date','place', 'visit']) [['date']].count()
df['#_of_visits'] = df2['date'].values