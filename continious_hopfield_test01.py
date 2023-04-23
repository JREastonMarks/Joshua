import numpy as np
from models.ContinuousHopfield import ContinuousHopfield

hf = ContinuousHopfield(16, beta = 8)



letterDNot = np.array([[ 5, 2, 3, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float64).transpose()
letterD = np.array([[ 20, 23, 23, 23, 20, 20, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25]], dtype=np.float64).transpose()
queryD  = np.array([[ 20, 23, 23, 23, 20, 20, 22, 22, 22, 22, 22, 20, 20, 20, 20, 20]], dtype=np.float64).transpose()

print(letterDNot.transpose())
print(letterD.transpose())
print(queryD.transpose())
print(np.array_equal(queryD, letterD))


hf.train(letterD)
hf.train(letterDNot)

result = hf.query(queryD, 10)

print(letterD.transpose())
print(result.transpose())

print(np.array_equal(result, queryD))
print(np.array_equal(result, letterD))
print(np.array_equal(result, letterDNot))