import pandas as pd
import Functions as fct

ethnicity_locality = pd.read_csv(filepath_or_buffer='./dataIN/Ethnicity.csv',
                                 index_col=0)
#print(ethnicity_locality, type (ethnicity_locality))

#Create pandas.DataFrame for storing the maping on counties
counties = pd.read_excel('./dataIN/CoduriRomania.xlsx', 'Localitati', index_col=0)

#print(counties, type(counties))



#we merge the ethnicity dataframe and the county dataframe via their index, hence left and right index is true
county_merge = ethnicity_locality.merge(right=counties, left_index =True, right_index = True)
print (county_merge)

#logic: merge, group by, filter by column
#df[specify list of columns]

#create a list of necessary columns
ethnicity_list = ethnicity_locality.columns.values[1:].tolist() #start with the second column to the end
#.tolist is very important because the ethnicity_locality is initially a ndarray and we can not add it to
#the county list
print (ethnicity_list)

county_list = ethnicity_list + ['County']
print(county_list)

county_aggr = county_merge[county_list].groupby(by='County').sum()
print(county_aggr)
#export the aggregation into a csv file
county_aggr.to_csv('./dataOUT/CountyEthnicity.csv')

regions = pd.read_excel('./dataIN/CoduriRomania.xlsx', 'Judete', index_col=0)
region_aggr = county_aggr.merge(right=regions, left_index =True, right_index = True)
print (region_aggr)
region_aggr.to_csv('./dataOUT/RegionEthnicity.csv')

macro_regions = pd.read_excel('./dataIN/CoduriRomania.xlsx', 'Regiuni', index_col=0)
macro_region_aggr = region_aggr.merge(right=macro_regions, on='Regiune')
print (macro_region_aggr)
macro_region_aggr.to_csv('./dataOUT/MacroRegionEthnicity.csv')

#call the dissimilarity index
result_dissimilarity = fct.dissimilarity_index(ethnicity_locality, ethnicity_list)
print(result_dissimilarity, type(result_dissimilarity))