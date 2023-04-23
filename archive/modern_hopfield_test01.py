import numpy as np
from models.ModernHopfield import ModernHopfield

hf = ModernHopfield(16)

# letterD = [ 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1]
# letterD = np.array([ [ -1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
# queryD  = np.array([ [ -1, -1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
letterD = np.array([[ -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
letterDNot = np.array([[ 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
queryD  = np.array([[ -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
# letterD = np.array([[ -3, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
# queryD  = np.array([[ -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
print(letterDNot)
print(letterD)
print(queryD)
print(np.array_equal(queryD, letterD))
hf.train(letterD)
hf.train(letterDNot)


result = hf.query(queryD, 10)


print(result.transpose())

print(np.array_equal(result.transpose(), queryD))
print(np.array_equal(result.transpose(), letterD))
print(np.array_equal(result.transpose(), letterDNot))