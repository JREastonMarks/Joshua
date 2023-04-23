from HRR.with_jax import normal, projection, binding, unbinding, cosine_similarity
import numpy as np

# from HRR import binding, unbinding, cosine_similarity
# import numpy as np

batch = 25
features = 25

x = projection(normal(shape=(batch, features), seed=0), axis=-1)
y = projection(normal(shape=(batch, features), seed=1), axis=-1)
z = projection(normal(shape=(batch, features), seed=2), axis=-1)
# x = np.mod(np.random.permutation(25*25).reshape(25,25),2)
# y = np.mod(np.random.permutation(25*25).reshape(25,25),2) 
# z = np.mod(np.random.permutation(25*25).reshape(25,25),2) 


a = binding(x, y, axis=-1)
b = binding(y, z, axis=-1)
c = a + b

y_prime1 = unbinding(a, x, axis=-1)
y_prime2 = unbinding(b, z, axis=-1)
y_prime3 = unbinding(c, x, axis=-1)
y_prime4 = unbinding(y_prime3, z, axis=-1)


score1 = cosine_similarity(y, y_prime1, axis=-1, keepdims=False)
score2 = cosine_similarity(y, y_prime2, axis=-1, keepdims=False)
score3 = cosine_similarity(y, y_prime3, axis=-1, keepdims=False)
score4 = cosine_similarity(y, y_prime4, axis=-1, keepdims=False)

# print(x)
# print(y)
# print(y_prime1)
# print(a)
# print('score1:', score1)
print('score1:', score1[0])
print('score2:', score2[0])
print('score3:', score3[0])
print('score4:', score4[0])