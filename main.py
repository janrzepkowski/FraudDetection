import pandas as pd


def main():
    df = pd.read_csv('Final Transactions.csv')
    print(df.shape)
    sample = df.sample(1000000)
    print(sample.shape)
    data = sample.get(['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'])
    print(data.head(20))
    print(data.shape)


if __name__ == '__main__':
    main()
