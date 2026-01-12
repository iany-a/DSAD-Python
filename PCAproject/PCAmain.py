import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def standardize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=0)
    return (X - means) / stds

class PCA:
    def __init__(self, X_std):
        self.X_std = X_std
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


def scree_plot(eigenvalues):
    plt.figure(figsize=(10, 6))
    comps = [f"C{i+1}" for i in range(len(eigenvalues))]
    plt.plot(comps, eigenvalues, marker="o")
    plt.axhline(y=1, color="red", linestyle="--")
    plt.title("Explained variance by the principal components")
    plt.xlabel("Principal components")
    plt.ylabel("Eigenvalues")
    plt.tight_layout()
    # plt.show()

table = pd.read_csv("./dataIN/Sleep_PCA.csv", index_col=0)
obs = table.index.values
vars_ = table.columns.values
X = table[vars_].values.astype(float)

X_std = standardize(X)
pd.DataFrame(X_std, index=obs, columns=vars_).to_csv("./dataOUT/X_std.csv")

pca_model = PCA(X_std)

alpha = pca_model.getAlpha()
scree_plot(alpha)
pd.DataFrame({"Eigenvalue": alpha},
             index=[f"C{i+1}" for i in range(len(alpha))]).to_csv("./dataOUT/Eigenvalues.csv")

C = pca_model.getComponents()
C_df = pd.DataFrame(C, index=obs, columns=[f"C{i+1}" for i in range(C.shape[1])])
C_df.to_csv("./dataOUT/PrincipalComponents.csv")

scores = pca_model.getScores()
scores_df = pd.DataFrame(scores, index=obs, columns=[f"C{i+1}" for i in range(scores.shape[1])])
scores_df.to_csv("./dataOUT/Scores.csv")

loadings = pca_model.a * np.sqrt(pca_model.alpha)
loadings_df = pd.DataFrame(loadings, index=vars_, columns=[f"C{i+1}" for i in range(loadings.shape[1])])
loadings_df.to_csv("./dataOUT/FactorLoadings.csv")

plt.figure(figsize=(10, 7))
sb.heatmap(np.round(loadings_df, 2), cmap="bwr", center=0, annot=True)
plt.title("Factor loadings (variables vs principal components)")
plt.tight_layout()
plt.show()

