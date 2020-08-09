# Import the standard libraries
import pandas as pd
import numpy as np

def main():
    # Visualize the different sheets of the excel file
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.ExcelFile(url)

    # todo Leisure
    # Data download
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.read_excel(url, sheet_name='FREIZEIT')

    # Change format of month column
    for index in df.index:
        item = str(df.MONAT.loc[index])
        df.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    df['DATE'] = pd.to_datetime(df['MONAT'])
    # Drop unused columns
    df_clean = df.drop(['MONATSZAHL', 'JAHR', 'MONAT'], axis=1)

    # Save the cleaned data as a csv file
    df_clean = df_clean[['AUSPRAEGUNG', 'WERT', 'DATE']]
    df_clean = df_clean[df_clean.WERT != np.NaN]
    df_clean.to_csv('../data/munich_visitors/munich-leisure-facilities.csv', index=False)

    # todo Cinema
    # Data download
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.read_excel(url, sheet_name='KINOS')

    # Change format of month column
    for index in df.index:
        item = str(df.MONAT.loc[index])
        df.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    df['DATE'] = pd.to_datetime(df['MONAT'])
    # Drop unused columns
    df_clean = df.drop(['MONATSZAHL', 'JAHR', 'MONAT'], axis=1)

    # Drop column "Auspraegung"
    df_clean = df_clean.drop(['AUSPRAEGUNG'], axis=1)

    # Save the cleaned data as a csv file
    df_clean = df_clean[['WERT', 'DATE']]
    df_clean = df_clean[df_clean.WERT != np.NaN]
    df_clean.to_csv('../data/munich_visitors/munich-cinemas.csv', index=False)

    # todo Museums
    # Data download
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.read_excel(url, sheet_name='MUSEEN')

    # Change format of month column
    for index in df.index:
        item = str(df.MONAT.loc[index])
        df.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    df['DATE'] = pd.to_datetime(df['MONAT'])
    # Drop unused columns
    df_clean = df.drop(['MONATSZAHL', 'JAHR', 'MONAT'], axis=1)

    for index in df_clean.index:
        if df_clean.DATE.loc[index] >= pd.to_datetime('2020-01-01'):
            if np.isnan(df_clean.WERT.loc[index]):
                # display(pd.DataFrame(df_clean.loc[index]))
                df_clean.WERT.loc[index] = 0
    print(df_clean[df_clean.DATE >= '2020-01-01'].copy().isna().sum())

    # Save the cleaned data as a csv file
    df_clean = df_clean[['AUSPRAEGUNG', 'WERT', 'DATE']]
    df_clean = df_clean[df_clean.WERT != np.NaN]
    df_clean.to_csv('../data/munich_visitors/munich-museums.csv', index=False)

    # todo orchestra
    # Data download
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.read_excel(url, sheet_name='ORCHESTER')

    # Change format of month column
    for index in df.index:
        item = str(df.MONAT.loc[index])
        df.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    df['DATE'] = pd.to_datetime(df['MONAT'])
    # Drop unused columns
    df_clean = df.drop(['JAHR', 'MONAT'], axis=1)

    for index in df_clean.index:
        if df_clean.DATE.loc[index] >= pd.to_datetime('2020-01-01'):
            if np.isnan(df_clean.WERT.loc[index]):
                # display(pd.DataFrame(df_clean.loc[index]))
                df_clean.WERT.loc[index] = 0
    print(df_clean[df_clean.DATE >= '2020-01-01'].copy().isna().sum())

    # Save the cleaned data as a csv file
    df_clean = df_clean[df_clean.MONATSZAHL == 'Besucher*innen']
    df_clean = df_clean.drop(['MONATSZAHL'], axis=1)
    df_clean = df_clean[['AUSPRAEGUNG', 'WERT', 'DATE']]
    df_clean = df_clean[df_clean.WERT != np.NaN]
    df_clean.to_csv('../data/munich_visitors/munich-orchestra.csv', index=False)

    # todo theater
    # Data download
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.read_excel(url, sheet_name='THEATER')

    # Change format of month column
    for index in df.index:
        item = str(df.MONAT.loc[index])
        df.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    df['DATE'] = pd.to_datetime(df['MONAT'])
    # Drop unused columns
    df_clean = df.drop(['JAHR', 'MONAT'], axis=1)

    for index in df_clean.index:
        if df_clean.DATE.loc[index] >= pd.to_datetime('2020-01-01'):
            if np.isnan(df_clean.WERT.loc[index]):
                # display(pd.DataFrame(df_clean.loc[index]))
                df_clean.WERT.loc[index] = 0
    print(df_clean[df_clean.DATE >= '2020-01-01'].copy().isna().sum())

    # Save the cleaned data as a csv file
    df_clean = df_clean[df_clean.MONATSZAHL == 'Besucher*innen']
    df_clean = df_clean.drop(['MONATSZAHL'], axis=1)
    df_clean = df_clean[['AUSPRAEGUNG', 'WERT', 'DATE']]
    df_clean = df_clean[df_clean.WERT != np.NaN]
    df_clean.to_csv('../data/munich_visitors/munich-theatre.csv', index=False)

    # todo tourism
    # Data download
    url = 'https://www.mstatistik-muenchen.de/monatszahlenmonitoring/export/xlsx/mzm_export_alle_monatszahlen.xlsx'
    df = pd.read_excel(url, sheet_name='TOURISMUS')

    # Change format of month column
    for index in df.index:
        item = str(df.MONAT.loc[index])
        df.MONAT.loc[index] = item[:4] + '-' + item[-2:]
    df['DATE'] = pd.to_datetime(df['MONAT'])
    # Drop unused columns
    df_clean = df.drop(['JAHR', 'MONAT'], axis=1)

    # Save the cleaned data as a csv file
    df_clean = df_clean[df_clean.MONATSZAHL == 'Gäste']
    df_clean = df_clean.drop(['MONATSZAHL'], axis=1)
    df_clean = df_clean[['AUSPRAEGUNG', 'WERT', 'DATE']]
    df_clean = df_clean[df_clean.WERT != np.NaN]
    df_clean.to_csv('../data/munich_visitors/munich-tourism.csv', index=False)

if __name__ == '__main__':
    main()