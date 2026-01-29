import pandas as pd
import numpy as np
import sklearn.cross_decomposition as skl
from matplotlib import pyplot as plt

#index_col = 0 takes first column SIRUTA code, merge on index
ind_df = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
#print(ind_df.head())

pop_df = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)

#print (pop_df.head())



#cerinta 1
#split value from each row on the no of inhabitants
#merge the tables and select the columns we need and the divide by inhabitants
merge_df = ind_df.merge(right=pop_df, left_index=True, right_index=True)
#print(merge_df)

#create list of columns with industries
#[1:] as we select all columns from Industries file
industry_list = ind_df.columns[1:].values.tolist()
#print(industry_list, type(industry_list))

def perCapita(df, vars, pop):
    row = df[vars] / df[pop]
    #print(row)
    rez = list(row)
    rez = [df['Localitate_x']] + rez
    return pd.Series(data=rez, index=['Localitate'] + vars)

#result_df = merge_df[['Localitate_x', 'Populatie'] + industry_list].apply(func=perCapita, axis=1, vars = industry_list, pop= 'Populatie')]
rezultat_df = merge_df[ [ 'Localitate_x' , 'Populatie' ] + industry_list].apply(func=perCapita, axis=1, vars=industry_list, pop='Populatie')
#siruta needs to be a column in the exported file!!!

rezultat_df.to_csv('./dataOUT/Cerinta1.csv')

#cerinta 2
df_1 = merge_df[industry_list + ['Judet']].groupby(by='Judet').sum()
#print(df_1)

def dominantIndustry(df): #only a dataframe as input
    row = df.values
    #print(row)
    maxCA = np.argmax(row) #this gives us the index of max value from that row
    return pd.Series(data=[df.index[maxCA], row[maxCA]], index = ['Industria dominanta', 'Cifra de afaceri'])



df_2 = df_1[industry_list].apply(func=dominantIndustry, axis=1)
#print(df_2)
df_2.to_csv('./dataOUT/Cerinta2.csv', index_label = 'County')

#cerinta_3
#split the dataset in 2 and save in 2 csv files
tabel = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)
#print(tabel)

#z_score = lambda df: (df- df.mean() / df.std(ddof=0))
def quick_std(df):
    return df.sub(df.mean()).div(df.std(ddof=0))

Xstd_df_quick = quick_std(tabel.iloc[:, :4])
Ystd_df_quick = quick_std(tabel.iloc[:, 4:])

Xstd_df_quick.to_csv('./dataOUT/Xscores.csv', index_label = 'Country')
Ystd_df_quick.to_csv('./dataOUT/Yscores.csv', index_label = 'Country')

def standardize(A):
    #expects a numpy.ndarray
    means = np.mean(a=A, axis=0) #column mean
    std_deviations = np.std(A, axis=0)
    return (A - means) / std_deviations

x_col = tabel.columns[0:4].values
print (x_col)
X = tabel[x_col].values
Xstd = standardize(X)
Xstd_df = pd.DataFrame(data=Xstd, index=tabel.index.values, columns = x_col)

Xstd_df.to_csv('./dataOUT/Xstd.csv', index_label = 'Country')

y_col = tabel.columns[4:].values
print (y_col)
Y = tabel[y_col].values
Ystd = standardize(Y)
Ystd_df = pd.DataFrame(data=Ystd, index=tabel.index.values, columns = y_col)

Ystd_df.to_csv('./dataOUT/Ystd.csv', index_label = 'Country')

#cerinta_4
n, p = Xstd_df_quick.shape
#print(n,p)
#skl.CCA = skl.CCA(n_components=2)
q = Ystd_df_quick.shape[1] #this returns the number of columns, only for Y used
#print(q)
# Extract the observation names (indices) from the original table
obs = tabel.index.values
#no of canonical pairs
m = min(p,q)
object_CCA = skl.CCA(n_components=m)
object_CCA.fit(X=Xstd_df_quick, y=Ystd_df_quick) #pay attention to X and y lettering
z, u = object_CCA.transform(X=Xstd_df_quick, y=Ystd_df_quick)
# print (z)
# print (u)

z_df = pd.DataFrame(data=z, index=obs, columns = x_col)
z_df.to_csv('./dataOUT/Xscores.csv', index_label = 'Country')
u_df = pd.DataFrame(data=u, index=obs, columns = y_col)
u_df.to_csv('./dataOUT/Yscores.csv', index_label = 'Country')


#cerinta 5
#4x4 matrix
Rxz = object_CCA.x_loadings_
#print(Rxz)
Rxz_df = pd.DataFrame(data=Rxz, index=x_col, columns = ('z'+str(j+1) for j in range (m)))
Rxz_df.to_csv('./dataOUT/Rxz.csv')

Ryu = object_CCA.y_loadings_
#print(Ryu)
Ryu_df = pd.DataFrame(data=Ryu, index=y_col, columns = ('y'+str(j+1) for j in range (m)))
Ryu_df.to_csv('./dataOUT/Ryu.csv')

#cerinta 6
#plot graph >> coordinates for 2 scatter plots overlapped
def biplot(z, u, title='Biplot', labelX='z1, u1', labelY='z2, u2', obs=None):
    plt.figure(num=title, figsize=(15,10))
    plt.title(label=title, fontsize=12, verticalalignment='top')
    plt.xlabel(xlabel=labelX, fontsize=10, verticalalignment='top')
    plt.ylabel(ylabel=labelY, fontsize=10, verticalalignment='top')
    plt.scatter(x=z[:,0], y=z[:, 1], color='Red', label='Set X')
    plt.scatter(x=u[:,0], y=u[:, 1], color='Blue', label = 'Set Y')
    if obs is not None:
        for i in range(len(obs)):
            plt.text(x=z[i,0], y=z[i,1], s=obs[i])
            plt.text(x=u[i, 0], y=u[i, 1], s=obs[i])
    plt.legend()

biplot(z=z, u=u, obs=obs)
plt.show()


