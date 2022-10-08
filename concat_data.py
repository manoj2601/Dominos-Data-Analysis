import pandas as pd 

df1 = pd.read_excel('./Bhopal_17_To_23_Dec.xlsx')
# l = df.to_dict('records')

df2 = pd.read_excel('./Bhopal_24_To_31_Dec.xlsx')

df3 = pd.concat([df1, df2])

df3 = df3.groupby(by=['STORE_ID', 'STORE_LATITUDE', 'STORE_LONGITUDE']).to_frame()
df3.to_excel('stores.xlsx')
print(df3)
# print(len(df1))
# print(len(df2))
# print(len(df3))