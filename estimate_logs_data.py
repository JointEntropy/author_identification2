import pandas as pd
import os


wiki_logs = dict(

)

loveread_logs_params = dict(
    filepath_or_buffer='data/loveread_classic_log.csv',
    sep=',',
    quotechar='/'
)


if __name__ == '__main__':
    print(os.getcwd())
    df = pd.read_csv(**loveread_logs_params)
    print(df.columns)
    print('Число авторов ', df['author'].value_counts().shape[0])
    print('Число фрагментов', df.shape[0])
    print('Число произведений', df['name'].value_counts().shape[0])

