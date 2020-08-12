# Import the standard libraries
import pandas as pd
import numpy as np
import datetime

def download_data(sheet_name):
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    return pd.read_excel(url, sheet_name=sheet_name)

def convert_time(data):
    # Change format of month column
    for index in data.index:
        item = str(data.MONAT.loc[index])
        data.MONAT.loc[index] = '01' + '/' + item[-2:] + '/' + item[:4]
        data.MONAT.loc[index] = datetime.datetime.strptime(data.MONAT.loc[index],'%d/%m/%Y').strftime('%Y/%m/%d')
    data = data.rename(columns={"MONAT": "DATE", "B": "c"})
    return data.drop(['MONATSZAHL', 'JAHR'], axis=1)

def main():
    sheets = ['FREIZEIT','KINOS','MUSEEN','ORCHESTER','THEATER','TOURISMUS']
    ret = []
    for sheet_name in sheets:
        print('Downloading data for: ' + sheet_name + '...')
        # Data download
        # TODO: Download could be accelerated by downloading the Excel file only once.
        df = download_data(sheet_name)

        df = convert_time(df)
        df_clean = df[['AUSPRAEGUNG', 'WERT', 'DATE']]
        df_clean = df_clean[df_clean.WERT != np.NaN]

        # Set set nan values from 2020 to zero
        for index in df_clean.index:
            if datetime.datetime.strptime(df_clean.DATE.loc[index], '%Y/%m/%d') >= datetime.datetime.strptime(
                    '2020/01/01', '%Y/%m/%d'):
                if np.isnan(df_clean.WERT.loc[index]):
                    df_clean.WERT.loc[index] = 0
            else:
                if np.isnan(df_clean.WERT.loc[index]):
                    df_clean.WERT.loc[index] = df_clean.WERT.mean()

        # Set date as index
        df_clean = df_clean.set_index('DATE')

        # Generate a compund feature table
        for item in df_clean.AUSPRAEGUNG.unique():
            name = item
            tmp = pd.DataFrame(df_clean[df_clean.AUSPRAEGUNG == name].copy().WERT)
            tmp = pd.DataFrame(tmp)
            ret.append(pd.DataFrame(tmp.rename(columns={'WERT': name})))

    # Concat the feature table
    df_clean = pd.concat(ret, axis=0, sort=True)
    df_clean = df_clean.groupby(['DATE'], sort=True).sum()
    df_clean = df_clean.reset_index()
    df_clean['DATE'] = [datetime.datetime.strptime(date, '%Y/%m/%d').strftime('%d/%b/%Y') for date in df_clean['DATE']]
    df_clean = df_clean.set_index('DATE')

    # Summing up some columns
    df_clean['Deutsches Museum - Museumsinsel'] = df_clean['Deutsches Museum - Museumsinsel'] + df_clean[
        'Deutsches Museum - Verkehrszentrum']
    df_clean = df_clean.drop(['Deutsches Museum - Verkehrszentrum'], axis=1)
    df_clean = df_clean.rename(columns={"Deutsches Museum - Museumsinsel": "Deutsches Museum"})
    # df_clean['Außenanlagen Olympiapark (Veranstaltungen)'] = df_clean['Außenanlagen Olympiapark (Veranstaltungen)'] + df_clean['Olympia-Eissportzentrum'] + df_clean['Olympiahalle'] + df_clean['Olympiaturm'] + df_clean['Kleine Olympiahalle']
    # df_clean = df_clean.drop(['Olympia-Eissportzentrum', 'Olympiahalle', 'Olympiaturm', 'Kleine Olympiahalle'],axis=1)
    # df_clean = df_clean.rename(columns={"Außenanlagen Olympiapark (Veranstaltungen)": "Olympiapark"})

    # Rename some columns for better clarity
    df_clean = df_clean.rename(columns={"insgesamt": "Kinos"})
    df_clean = df_clean.rename(columns={"Prinzregententheater (Großes Haus)": "Prinzregententheater"})
    df_clean = df_clean.rename(columns={"Ausland": "Ausland (Tourismus)"})
    df_clean = df_clean.rename(columns={"Inland": "Inland (Tourismus)"})
    df_clean = df_clean.rename(columns={"Außenanlagen Olympiapark (Veranstaltungen)": "Olympiapark"})

    # Reset index
    df_clean = df_clean.reset_index()

    #Save file
    df_clean.to_csv('../data/munich_visitors/munich_visitors.csv', index=False)

if __name__ == '__main__':
    main()