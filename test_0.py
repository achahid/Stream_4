





#%%

import pandas as pd
import openpyxl


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#%%
keywords_df  = pd.read_csv('D:\\Tutorials\\Streamlit\\INPUT_DATA\\locatie_nl.csv', sep=',',encoding='latin-1')
df = keywords_df[:2000].copy()
df["Keyword"] = df["Keyword"].astype(str)


#%%





