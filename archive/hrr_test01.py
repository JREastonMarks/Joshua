# from HRR.with_pytorch import normal, projection, binding, unbinding, cosine_similarity
# from HRR import binding, unbinding, cosine_similarity
import numpy as np

batch = 32
features = 256

x = projection(normal(shape=(batch, features), seed=0), dim=-1)
y = projection(normal(shape=(batch, features), seed=1), dim=-1)
z = projection(normal(shape=(batch, features), seed=2), dim=-1)
# x = np.mod(np.random.permutation(4*4).reshape(4,4),2)
# y = np.mod(np.random.permutation(4*4).reshape(4,4),2) 
# z = np.mod(np.random.permutation(4*4).reshape(4,4),2) 

a = binding(x, y, dim=-1)
b = binding(y, z, dim=-1)
c = a + b

y_prime1 = unbinding(a, x, dim=-1)
y_prime2 = unbinding(b, z, dim=-1)
y_prime3 = unbinding(c, x, dim=-1)
y_prime4 = unbinding(y_prime3, z, dim=-1)


score1 = cosine_similarity(y, y_prime1, dim=-1, keepdim=False)
score2 = cosine_similarity(y, y_prime2, dim=-1, keepdim=False)
score3 = cosine_similarity(y, y_prime3, dim=-1, keepdim=False)
score4 = cosine_similarity(y, y_prime4, dim=-1, keepdim=False)

print(x)
print(y)
print(a)
print('score1:', score1[0])
print('score2:', score2[0])
print('score3:', score3[0])
print('score4:', score4[0])