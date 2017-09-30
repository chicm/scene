import pandas as pd 
import numpy as np

#src_csv = '/home/chicm/ml/kgdata/species/results_tta/tta_25.csv'
src_csv = '/home/chicm/ml/kgdata/species/results_tta/99484/tta15.csv'
#tgt_csv = '/home/chicm/ml/kgdata/species/results_tta/tta_25_bin.csv'
tgt_csv = '/home/chicm/ml/kgdata/species/results_tta/99484/tta15_2.csv'

df1 = pd.read_csv(src_csv)
df1['invasive'] = df1['invasive'].apply(lambda x: 0.999999 if x > 0.96 else x)
df1['invasive'] = df1['invasive'].apply(lambda x: 0.000001 if x < 0.04 else x)

df1.to_csv(tgt_csv, index=False)

#df2 = pd.read_csv(src_csv)



#df2['invasive'] = (df1.values[:, 1] >= 0.5).astype(np.uint8)
#df2.to_csv(tgt_csv, index=False)
