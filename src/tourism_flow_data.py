import pandas as pd


def preprocessing(df):
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Trip type: ': ''}, regex=True)
    df['date'] = [datetime.strptime(date, '%B %Y') for date in df['date']]
    df = df.sort_values(by='date', ascending=False, inplace=False, ignore_index=True)
    df = df.set_index('date')

    return df


def clustering_process(df):
    df[['city', 'country', 'extra']] = df['visitor_origin'].str.split(', ', expand=True, n=2)
    df = df.drop(['rating', 'title', 'text'], axis=1)
    return df


def feature_extraction(df):
    df = preprocessing(df)
    df = clustering_process(df)
    visitors_by_country = df.groupby('country').count().sort_values('visit', ascending=True)
    type_of_visitors = df.groupby('visit').count().sort_values('country', ascending=True)
    visitors_by_city = df.groupby('city').count().sort_values('visit', ascending=True)

    return visitors_by_country, type_of_visitors, visitors_by_city


def eu_countries(visitors_by_country):
    visitors_by_country["Non EU"] = 0
    for i in range(len(visitors_by_country)):
        if not (visitors_by_country.index[i] in EU_countries):
            visitors_by_country["Non EU"][i] = int(1)
    return visitors_by_country


def get_visitors(visitors_by_country, visitors_by_city):
    visitors_from_munich = visitors_by_city['visitor_origin']['Munich']
    visitors_outside_munich = visitors_by_country['visitor_origin']['Germany'] - visitors_by_city['visitor_origin'][
        'Munich']
    visitors_outside_eu = visitors_by_country.groupby('Non EU').sum()['visitor_origin'][1]
    visitors_from_eu = visitors_by_country.groupby('Non EU').sum()['visitor_origin'][0] - \
                       visitors_by_country['visitor_origin']['Germany']
    return visitors_from_munich, visitors_outside_munich, visitors_outside_eu, visitors_from_eu


def get_file(path):
    file_names = []
    data = pd.DataFrame()
    names = glob.glob(path)
    for i in range(len(names)):
        df = pd.read_csv(names[i], header=0, squeeze=True)
        file_name = os.path.basename(names[i])
        file_name = file_name.split('.')[0]
        file_names.append(file_name)
        visitors_by_country, type_of_visitors, visitors_by_city = feature_extraction(df, file_name)
        visitors_by_country = eu_countries(visitors_by_country)
        visitors_from_munich, visitors_outside_munich, visitors_outside_eu, visitors_from_eu = get_visitors(
            visitors_by_country, visitors_by_city)
        type_of_visitors['visitors_from_munich'] = visitors_from_munich
        type_of_visitors['visitors_outside_munich'] = visitors_outside_munich
        type_of_visitors['visitors_outside_eu'] = visitors_outside_eu
        type_of_visitors['visitors_from_eu'] = visitors_from_eu
        type_of_visitors['attraction_name'] = file_name
        data = data.append(type_of_visitors)

        print(i)
    data.reset_index()
    data.set_index('attraction_name', inplace=True)
    data.to_csv('k_means_data.csv')

    return data


data = get_file(path)
