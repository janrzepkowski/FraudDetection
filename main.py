import pandas as pd
import zipfile
import os


def main():
    if not os.path.exists('Final Transactions.csv'):
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall()

    df = pd.read_csv('Final Transactions.csv')
    print(df.shape)
    sample = df.sample(1000000)
    print(sample.shape)
    data = sample.get(['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'])
    print(data.head(20))
    print(data.shape)


if __name__ == '__main__':
    main()
