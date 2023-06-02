import pandas as pd
nb_sbjs = 7
df = pd.DataFrame()
for sbj in range(nb_sbjs):

    df_csv = pd.read_csv(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230322\val_pred_sbj_{sbj+1}.csv', \
                     index_col = False, header= None, delimiter = ',')
    # print(df_csv)
    # Remove the first row and column
    # df_csv = df_csv.drop(df_csv.index[0])
    # df_csv = df_csv.drop(df_csv.columns[0], axis=1)
    print(df_csv)
    df = pd.concat([df, df_csv], ignore_index= True)
    # print(len(df_csv))
    print(df)
df.to_csv(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230322\val_pred_7_sbj.csv', index = None, header = None)
print(df)