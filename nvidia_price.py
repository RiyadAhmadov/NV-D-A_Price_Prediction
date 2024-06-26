import pandas as pd

df = pd.read_excel('nvidia_price.xlsx')
df1 = pd.DataFrame(pd.date_range(df['Date'].min(),df['Date'].max()),columns = ['Date'])
df1 = df1.merge(df, on = 'Date',how = 'left')
df1.bfill(axis = 'rows',inplace = True)
df1.to_excel('nvidia_report.xlsx',index = False)