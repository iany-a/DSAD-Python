import pandas as pd
import numpy as np
import graphicsHCA as g
import utilsHCA as utl
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl


# 1. Load the data
fileName = './dataIN/Sleep_health_and_lifestyle_dataset.csv'
table = pd.read_csv(fileName, index_col=0) # Using Person ID as index

# --- START PRE-PROCESSING BLOCK ---
# A. Fix Blood Pressure (Split '126/83' into two columns)
table[['Systolic', 'Diastolic']] = table['Blood Pressure'].str.split('/', expand=True).astype(float)
table = table.drop(columns=['Blood Pressure'])

# B. Encode BMI (Ordinal)
bmi_map = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}
table['BMI Category'] = table['BMI Category'].map(bmi_map)

# C. One-Hot Encode Occupation and Gender
# We do this before setting 'vars' because it creates new columns
table = pd.get_dummies(table, columns=['Occupation', 'Gender'], drop_first=True)

# D. Handle Sleep Disorder (Target Label)
# It's better to keep this out of the 'X' matrix so you can see if the clusters
# actually found the disorders naturally.
sleep_labels = table['Sleep Disorder'].fillna('None')
table = table.drop(columns=['Sleep Disorder'])
# --- END PRE-PROCESSING BLOCK ---

# Now the rest of your professor's code will work perfectly
obs = table.index.values
vars = table.columns.values # All columns are now numeric
n = len(obs)
m = len(vars)

table_numeric = table.apply(pd.to_numeric, errors='coerce')

table_nda = table_numeric.values.astype(np.float64)
X = utl.replace_na(table_nda)
X_std = utl.standardise(X)
X_std_df = pd.DataFrame(data=X_std, index=obs, columns=vars)
X_std_df.to_csv('./dataOUT/X_std.csv')

# create a list of clustering methods
methods = list(hclust._LINKAGE_METHODS)
print(methods, type(methods))

# create a list of metrics
metrics = hdist._METRICS_NAMES
print(metrics, type(metrics))

# determine the cluster of observations
h_1 = hclust.linkage(y=X_std, method='single', metric='cityblock')
print(h_1, type(h_1))
# compute the threshold, the junction at which the partition of maximum
# stability occurs, and the maximum number of junctions
threshold, j, k = utl.threshold(h_1)

# create the dendrogram graphic
g.dendrogram(h=h_1, labels=obs,
    title="Hierarchical classification - method='single', metric='cityblock'",
    threshold=threshold, colors=None)
# g.show()

# determine the cluster of variables
h_2 = hclust.linkage(y=X_std.T, method='complete', metric='correlation')
print(h_2, type(h_2))
# compute the threshold, the junction at which the partition of maximum
# stability occurs, and the maximum number of junctions
threshold, j, k = utl.threshold(h_2)

# create the dendrogram graphic
g.dendrogram(h=h_2, labels=vars,
    title="Hierarchical classification - method='median', metric='correlation'",
    threshold=threshold, colors=None)


# 1. Get the cluster codes for each person (observations)
# We use h_1 (the hierarchy for observations) and k (the stability threshold)
cluster_names, cluster_codes = utl.cluster_distribution(h_1, k)

# 2. Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Actual_Disorder': sleep_labels.values,
    'Cluster_Assigned': cluster_names
})

# 3. Create the Summary (Cross-tabulation)
summary_table = pd.crosstab(comparison_df['Cluster_Assigned'],
                            comparison_df['Actual_Disorder'])

print("--- Clustering Validation Table ---")
print(summary_table)

# 4. Save this to CSV for your project report
summary_table.to_csv('./dataOUT/cluster_validation.csv')


g.show()
# TODO
