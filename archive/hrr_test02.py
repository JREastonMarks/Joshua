import numpy as np


def circular_convolution(x, y):
    """A fast version of the circular convolution."""
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    z = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
    if np.ndim(z) == 1:
        z = z[None, :]
    return z

def cosine_similarity(x, y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return np.sum(norm_x * norm_y)

# def cosine_similarity(A, B):
    # return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def involution(x):
    """Involution operator."""
    if np.ndim(x) == 1:
        x = x[None, :]
    return np.concatenate([x[:, None, 0], x[:, -1:0:-1]], 1)


def circular_correlation(x, y):
    """Circular correlation is the inverse of circular convolution."""
    return circular_convolution(involution(x), y)


def decode(x, y):
    """Simple renaming."""
    return circular_correlation(x, y)

x = np.mod(np.random.permutation(4*4).reshape(4,4),2)
y = np.mod(np.random.permutation(4*4).reshape(4,4),2) 
z = np.mod(np.random.permutation(4*4).reshape(4,4),2) 

a = circular_convolution(x, y)
b = circular_convolution(y, z)
c = a + b

y_prime1 = decode(a, x)
y_prime2 = decode(b, z)
y_prime3 = decode(c, x)
y_prime4 = decode(y_prime3, z)




score1 = cosine_similarity(y, y_prime1)# , dim=-1, keepdim=False)
score2 = cosine_similarity(y, y_prime2)# , dim=-1, keepdim=False)
score3 = cosine_similarity(y, y_prime3)# , dim=-1, keepdim=False)
score4 = cosine_similarity(y, y_prime4)# , dim=-1, keepdim=False)

print(y)
print(y_prime1)
# print(y)
# print(a)
# print('score1:', score1)
# print('score2:', score2)
# print('score3:', score3)
# print('score4:', score4)