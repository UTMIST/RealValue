import pandas as pd
import numpy as np

df = pd.read_fwf('toronto_dataset/HousesInfo.txt')

new_df=pd.DataFrame(columns=['Bedrooms','Bathrooms','SqFt','ZipCode','Price'])

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

for index,row in df.iterrows():
    print(row.tolist(),index)
    final=row.tolist()[0].split()
    final = list(map(int, final))
    new_df.loc[index]=final

new_df = new_df.drop('ZipCode', 1)

print(new_df.head())
price_max=new_df['Price'].max()
sqft_max=new_df['SqFt'].max()

continuous_feats=['SqFt','Price']
categorical_feats=['Bedrooms','Bathrooms']
for i, feature in enumerate(continuous_feats):
    new_df[feature]=(new_df[feature]-new_df[feature].min())/(new_df[feature].max()-new_df[feature].min())
cts_data=new_df[continuous_feats].values

for i, feature in enumerate(categorical_feats):
    temp_column=pd.get_dummies(new_df[feature]).to_numpy()
    if i==0:
        cat_onehot=temp_column
    else:
        cat_onehot=np.concatenate((cat_onehot,temp_column),axis=1)
final_stats_array= np.concatenate([cat_onehot,cts_data], axis=1)
final_x_array=final_stats_array[:,:-1]
final_price_array= final_stats_array[:,-1]

print(final_x_array)
print(final_price_array)
