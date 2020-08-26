# Import the standard libraries
import pandas as pd
import json
import requests

def download_data(path='../data/covid_19_data/rki/RKI_COVID19.json'):
    url = 'https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.geojson'
    # Data download
    param = dict()
    resp = requests.get(url=url, params=param)
    data = resp.json()
    # Save data as a .json file
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    print("RKI data downloaded and saved to: "+path)

def convert_json(path = '../data/covid_19_data/rki/RKI_COVID19.json'):
    # Load .json file
    with open(path) as data_file:
        data = json.load(data_file)
    print("Read file: " + path)
    # Compile json to dataframe
    df = pd.json_normalize(data, 'features')
    # Drop unnecessary columns
    df = df.drop(['type', 'geometry'], axis=1)
    # Rename columns
    df.columns = [col.replace('properties.', '') for col in df.columns]
    return df

def data_cleaning(data, district_name, dropAnzahlTodesfall=True, dropAltersgruppe=True, dropGeschlecht=True):
    # Clear not used data
    df_clean = data.drop(['ObjectId','IdBundesland','NeuerFall','NeuerTodesfall','NeuGenesen','AnzahlGenesen','IstErkrankungsbeginn'], axis=1)
    # Delete not not used columns
    if dropAnzahlTodesfall:
        df_clean = df_clean.drop(['AnzahlTodesfall'], axis=1)
    if dropAltersgruppe:
        df_clean = df_clean.drop(['Altersgruppe'], axis=1)
    if dropGeschlecht:
        df_clean = df_clean.drop(['Geschlecht'], axis=1)

    # Select data of one district
    district = df_clean[df_clean.Landkreis == district_name].copy()
    # Change time format
    district['Refdatum'] = pd.to_datetime(district['Refdatum'])
    # Summing up the COVID-19 cases
    district = district.groupby(by=['Refdatum']).sum()
    # Sort the data
    district = district.sort_values(by=['Refdatum'])
    print("Data cleaning ended")
    return district

def main():
    data = download_data()
    print("Converting json to csv")
    df = convert_json()
    print("Data cleaning ...")
    district_name = 'SK München'
    df = data_cleaning(df, district_name)

    # Replace special characters
    district_name = district_name.replace("ü", "ue").replace(" ", "_")
    # Save cases of one district in a csv file
    file_name = '../data/covid_19_data/rki/COVID_19_Cases_' + district_name + '.csv'
    df.to_csv(file_name, index=True, encoding='utf-8')
    print("Saved cleaned data to: " + file_name)

if __name__ == '__main__':
    main()