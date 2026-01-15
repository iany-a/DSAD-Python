import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import fcluster
import graphicsHCA as g
import utils as utl
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sb

#HCA

fileName = './dataIN/Sleep_health_and_lifestyle_dataset.csv'

df = pd.read_csv(fileName)

# --- STEP 1: Process Complex Values (Blood Pressure) ---
# Split '126/83' into two numeric columns
df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop('Blood Pressure', axis=1, inplace=True)

# --- STEP 2: Ordinal Encoding (BMI Category) ---
# Mapping categories to numbers based on logical order
bmi_mapping = {
    'Underweight' : 0,
    'Normal': 1, 
    'Normal Weight': 1, 
    'Overweight': 2, 
    'Obese': 3
}
df['BMI Category'] = df['BMI Category'].map(bmi_mapping)

# --- STEP 3: One-Hot Encoding (Nominal Values) ---
# Turning Gender, Occupation, and Sleep Disorder into binary columns
df = pd.get_dummies(df, columns=['Gender', 'Occupation', 'Sleep Disorder'], dtype=int)

# --- STEP 4: Standardization (Min-Max Scaling) ---
# We exclude 'Person ID' from scaling as it is just an identifier
cols_to_scale = df.columns.difference(['Person ID'])
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Optional: Set Person ID as the index
df.set_index('Person ID', inplace=True)




obs = df.index.values
print(obs, type(obs), obs.shape)
n = len(obs)
print('No. of observations:', n)

vars = df.columns.values
print(vars, type(vars), vars.shape)
m = len(vars)
print('No. of variables:', m)

df_nda = df[vars].values
print(type(df_nda), df_nda.shape)

X = utl.replace_na(df_nda)
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
h_1 = hclust.linkage(y=X_std, method='ward', metric='euclidean')
print(h_1, type(h_1))
# compute the threshold, the junction at which the partition of maximum
# stability occurs, and the maximum number of junctions
threshold, j, k = utl.threshold(h_1)

# create the dendrogram graphic
g.dendrogram(h=h_1, labels=obs,
    title="Hierarchical classification - method='ward', metric='euclidean'",
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


# 1. Generate cluster assignments
# 'threshold' is around 52.1294 based on ward plot
clusters = fcluster(h_1, 52.1294, criterion='distance')
df['Cluster'] = clusters

cluster_profile = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Sleep Duration': 'mean',
    'Quality of Sleep': 'mean',
    'Stress Level': 'mean',
    'Heart Rate': 'mean',
    'Daily Steps': 'mean'
}).round(2)

print(cluster_profile)

#PCA
#This class implements PCA using the covariance matrix and eigen decomposition
class PCA:
    def __init__(self, X_std):
        self.X_std = X_std

        # Compute covariance matrix
        self.cov = np.cov(self.X_std, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.alpha = eigenvalues[idx]
        self.a = eigenvectors[:, idx]

        self.C = self.X_std @ self.a

    def getAlpha(self):
        return self.alpha

    def getComponents(self):
        return self.C

    def getScores(self):
        return self.C / np.sqrt(self.alpha)

#Data Loading
table = pd.read_csv("./dataIN/Sleep_PCA.csv", index_col=0)

# Extraction of observation and variable names
obs = table.index.values
vars_ = table.columns.values
X2 = table[vars_].values.astype(float)
X2_std = utl.standardise(X2)

pd.DataFrame(X2_std, index=obs, columns=vars_).to_csv("./dataOUT/X2_std.csv")

pca_model = PCA(X2_std)

alpha = pca_model.getAlpha()
utl.scree_plot(alpha)

pd.DataFrame({"Eigenvalue": alpha},
             index=[f"C{i+1}" for i in range(len(alpha))]).to_csv("./dataOUT/Eigenvalues.csv")
C = pca_model.getComponents()
C_df = pd.DataFrame(C, index=obs, columns=[f"C{i+1}" for i in range(C.shape[1])])
C_df.to_csv("./dataOUT/PrincipalComponents.csv")

scores = pca_model.getScores()
scores_df = pd.DataFrame(scores, index=obs, columns=[f"C{i+1}" for i in range(scores.shape[1])])
scores_df.to_csv("./dataOUT/Scores.csv")

#Computation and visualization of factor loadings
loadings = pca_model.a * np.sqrt(pca_model.alpha)
loadings_df = pd.DataFrame(loadings, index=vars_, columns=[f"C{i+1}" for i in range(loadings.shape[1])])
loadings_df.to_csv("./dataOUT/FactorLoadings.csv")

plt.figure(figsize=(10, 7))
sb.heatmap(np.round(loadings_df, 2), cmap="bwr", center=0, annot=True)
plt.title("Factor loadings")
plt.tight_layout()
plt.show()


g.show()
