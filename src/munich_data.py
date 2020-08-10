# Import the standard libraries
import pandas as pd
import numpy as np

def download_data(sheet_name):
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    return pd.read_excel(url, sheet_name=sheet_name)

def convert_time(data):
    # Change format of month column
    # TODO: Check how to get rid of the warning in cell locating
    for index in data.index:
        item = str(data.MONAT.loc[index])
        data.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    data['DATE'] = pd.to_datetime(data['MONAT'])
    # Drop unused columns
    return data.drop(['MONATSZAHL', 'JAHR', 'MONAT'], axis=1)

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

        # Set date as index
        df_clean = df_clean.set_index('DATE')

        # Set set nan values from 2020 to zero
        for index in df_clean.index:
            if df_clean.DATE.loc[index] >= pd.to_datetime('2020-01-01'):
                if np.isnan(df_clean.WERT.loc[index]):
                    df_clean.WERT.loc[index] = 0
            else:
                if np.isnan(df_clean.WERT.loc[index]):
                    df_clean.WERT.loc[index] = df_clean.WERT.mean()

        # Generate a compund feature table
        for item in df_clean.AUSPRAEGUNG.unique():
            name = item
            tmp = pd.DataFrame(df_clean[df_clean.AUSPRAEGUNG == name].copy().WERT)
            tmp = pd.DataFrame(tmp)
            ret.append(pd.DataFrame(tmp.rename(columns={'WERT': name})))

    # Concat the feature table
    df_clean = pd.concat(ret, axis=0, sort=True)
    df_clean = df_clean.reset_index()
    df_clean = df_clean.groupby(['DATE']).sum()

    # Rename some columns for better clarity
    # TODO: Check why the column is not renamed
    df_clean = df_clean.rename(columns={"insgesamt": "Kinos"})

    # Reset index
    df_clean = df_clean.reset_index()

    #Save file
    df_clean.to_csv('../data/munich_visitors/Features.csv', index=False)

if __name__ == '__main__':
    main()