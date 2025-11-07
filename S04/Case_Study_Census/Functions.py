import itertools
import pandas as pd
import numpy as np
from pyparsing import results


def dissimilarity_index(df, cols):
    df_numeric = df.select_dtypes(include=[np.number])
    results = []
    for group1, group2 in itertools.combinations(df_numeric, 2, ):
        #x = df[cols].values
        #print(x, type(x))
        g1=df_numeric[group1].values
        print(g1)
        g2=df_numeric[group2].values
        total_g1=g1.sum()
        total_g2=g2.sum()
        if total_g1 == 0 or total_g2 == 0:
            d = np.nan  # or D = 0 if you prefer
        else:
            d = 0.5 * np.sum(np.abs((g1 / total_g1) - (g2 / total_g2)))
        results.append({
            'Ethnicity_1': group1,
            'Ethnicity_2': group2,
            'Dissimilarity_Index': d
        })

    return pd.DataFrame(results)




