import pandas as pd
'''
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
result.to_excel('TorontoDataset_new.xlsx')'''



#df = pd.read_csv('../USZIPCodes202103.csv', delimiter = ",")

#temp_try = pd.concat([main_dataframe, main_txt], axis=1, join="inner")

df = pd.read_csv('../USZIPCodes202103.csv', delimiter = ",")
main_dataframe = df.loc[:,["Zip Code","ZipLatitude","ZipLongitude"]]
main_dataframe = main_dataframe.rename(columns={"Zip Code": "Zip Code", "ZipLatitude": "Lat", "ZipLongitude":"Long"})

main_txt = pd.read_csv('../cali_toronto_raw_dataset/HousesInfo.txt', delimiter = " ",names=['Bedrooms','Bathrooms','SqFt','Zip Code','Price'])

main_txt = main_txt.reset_index().rename({'index':'index1'}, axis = 'columns')
print(main_dataframe)
print(main_txt)

temp_try = main_txt.join(main_dataframe.set_index('Zip Code'), on = 'Zip Code')
print(temp_try)
temp_try = temp_try.drop_duplicates()
temp_try = temp_try[['Bedrooms','Bathrooms','SqFt','Price','Lat','Long']]

long_max, long_min = temp_try['Long'].max(), temp_try['Long'].min()
lat_max, lat_min = temp_try['Lat'].max(), temp_try['Lat'].min()


temp_try['Long'] = (temp_try['Long'] - long_min) / (long_max - long_min)

temp_try['Lat'] = (temp_try['Lat'] - lat_min) / (lat_max - lat_min)



temp_try.to_excel('../raw_dataset_true.xlsx')

blah = temp_try['Lat'].isnull()
print('---------')
print(blah)
print(blah.sum())
print('---------')

print(temp_try)
