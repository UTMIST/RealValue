import pandas as pd

toronto_dataset = pd.read_excel('TorontoDataset.xlsx')
toronto_dataset_2 = toronto_dataset.copy()
toronto_dataset = toronto_dataset.loc[:,["Zip Code"]]

df = pd.read_csv('CA.txt', delimiter = "\t")
main_dataframe = df.loc[:,["T0A","54.766","-111.7174"]]
main_dataframe = main_dataframe.rename(columns={"T0A": "Zip Code", "54.766": "Long", "-111.7174":"Lat"})
print(toronto_dataset.columns)
print(toronto_dataset)

main_dataframe = main_dataframe.set_index('Zip Code')
toronto_dataset = toronto_dataset.set_index('Zip Code')

result = pd.concat([main_dataframe, toronto_dataset], axis=1, join="inner")
#result = result.drop_duplicates()

long_max, long_min = result['Long'].max(), result['Long'].min()
lat_max, lat_min = result['Lat'].max(), result['Lat'].min()

result['Long'] = (result['Long'] - long_min) / (long_max - long_min)

result['Lat'] = (result['Lat'] - lat_min) / (lat_max - lat_min)


toronto_dataset_2 = toronto_dataset_2.merge(result, how='inner', on='Zip Code')
toronto_dataset_2 = toronto_dataset_2.drop_duplicates()
result = pd.DataFrame(toronto_dataset_2.reset_index(drop=True))

result = result.drop(columns = ["Zip Code"])
result.to_excel('TorontoDataset_new.xlsx')
