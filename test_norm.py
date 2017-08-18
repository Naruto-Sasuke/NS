from sklearn import preprocessing
import numpy as np

a = np.arange(15).reshape(3, 5)
print(a)

print(preprocessing.normalize(a,norm='l2'))