import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage


natLocMov = pd.read_csv('dataIN/NatLocMovements.csv', index_col=0)
popLoc = pd.read_csv('dataIN/PopulationLoc.csv', index_col=0)
#req1 - compute rate of natural increase as nat_inc = b_rate - d_rate
#b_rate = b / 1000
#d_rate = d / 1000
merge_df = natLocMov.merge(right=popLoc, right_index=True, left_index=True)
#merge_df.to_csv('dataOUT/Merge_df.csv', index=False)
print(merge_df.columns.to_list())
l_rate = (merge_df['LiveBirths'] / merge_df['Population'])*1000
#print (l_rate.head())
d_rate = (merge_df[['Deceased', 'DeceasedUnder1Year', 'StillBirths']].sum(axis=1) / merge_df['Population'])*1000
print(d_rate.head())
natural_increase = l_rate-d_rate
result_1 = pd.DataFrame({
    'Natural Increase': natural_increase,
    'County Code': merge_df['CountyCode']
})
print(result_1)
result_1.to_csv('dataOUT/Request_1.csv', index=False)

#req2 - for each county, compute the localities where rates are the highest

indicators = ['Marriages', 'Deceased', 'DeceasedUnder1Year', 'Divorces', 'StillBirths', 'LiveBirths']
temp_list = [] #dataframe

ratesDf = (merge_df.iloc[:,1:7].div(merge_df['Population'], axis=0)) * 1000
ratesDf.insert(0, 'County Code', merge_df['CountyCode'])
ratesDf.insert(1, 'City', merge_df['City'])
print (ratesDf.columns.to_list())
for county in ratesDf['County Code'].unique():
    county_group = ratesDf[ratesDf['County Code'] == county]
    row = {'County': county} #dictionary
    for col in indicators:
        max_index = county_group[col].idxmax()
        winning_locality = ratesDf.loc[max_index, 'City']
        row[col] = winning_locality #dictionary
    temp_list.append(row)

result_2 = pd.DataFrame(temp_list)

print (result_2)
result_2.to_csv('dataOUT/Request_2.csv', index=False)

#req3 - Standardize + HCA

dset = pd.read_csv('dataIN/DataSet_34.csv', index_col=0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dset)
scaled_df = pd.DataFrame(scaled_data, columns=dset.columns, index=dset.index)
scaled_df.to_csv('dataOUT/Xstd.csv', index=False)
Z = linkage(scaled_df, method='ward', metric='euclidean')
clusters = fcluster(Z, t=5, criterion='maxclust')
dset['Cluster_ID'] = clusters
print(dset.groupby('Cluster_ID').mean())
print (dset['Cluster_ID'].head())
plt.figure(figsize=(10,5))
dendrogram(Z, labels=scaled_df.index)
plt.title('HCA Dendrogram')
plt.show()

std_cov_matrix = scaled_df.cov()
std_cov_matrix.to_csv('dataOUT/StdCov.csv', index=False)


