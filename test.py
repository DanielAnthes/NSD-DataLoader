import numpy as np
from math import ceil
import pandas as pd 

a = pd.DataFrame({"a": [2,3,1], "b": [1,2,3]})
b = pd.DataFrame({"a": [3, 2, 1, 4], "c": [1, 2, 3, 4]})

c = a.merge(b, on="a")

d = np.array([1,2,3,4])

np.where(d==3)