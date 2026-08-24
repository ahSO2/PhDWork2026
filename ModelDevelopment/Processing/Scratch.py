from scipy.stats import pearsonr
import numpy as np

a1 = np.array([1, 2, 3, 4])
a2 = np.array([np.nan, np.nan, np.nan, np.nan])
a3 = np.empty(shape=(4))

print(pearsonr(a1, a3))