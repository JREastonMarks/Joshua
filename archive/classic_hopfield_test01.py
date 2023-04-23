import numpy as np
from models.Hopfield import Hopfield

hf = Hopfield(16)

# letterD = [ 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1]
# letterD = np.array([ [ -1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
# queryD  = np.array([ [ -1, -1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
letterD = np.array([[ -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
queryD  = np.array([[ -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])



hf.train(letterD.transpose())


result = hf.query(queryD.transpose(), 10)

print(letterD)
print(queryD)
print(result.transpose())
print(np.equal(queryD, letterD))
print(np.equal(result.transpose(), letterD))