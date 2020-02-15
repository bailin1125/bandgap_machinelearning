import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
print("now let's do the ridge  model")
# 这个是简单的线性回归的项目
# 首先开始载入数据
# first set the  X is the hilbert matrix
X = 1. /(np.arange(1, 11)+np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
# compute the paths
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel("alpha")
plt.ylabel('weights')
plt.title("ridge coeffcients as a function of the reularization")
plt.axis('tight')
plt.show()
