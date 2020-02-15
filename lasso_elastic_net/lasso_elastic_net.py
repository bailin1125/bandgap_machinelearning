import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
print("this is the lasso method exeplaniton")

np.random.seed(42)
n_samples,n_features=50,200
X=np.random.randn(n_samples,n_features)
coef=3*np.random.randn(n_features)
inds=np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]]=0 #稀疏化系数

y=np.dot(X,coef)
#add noise
y+=0.01*np.random.normal(size=n_samples)

#put the data as the train and test data
n_samples=X.shape[0]
X_train,y_train=X[:n_samples//2],y[:n_samples//2]
X_test,y_test=X[n_samples//2:],y[n_samples//2:]

#开始进行lasso方法
from sklearn.linear_model import Lasso
alpha=0.1
lasso=Lasso(alpha=alpha)
y_pred_lasso=lasso.fit(X_train,y_train).predict(X_test)
r2_score_lasso=r2_score(y_test,y_pred_lasso)
print("lasso")
print("r^2的值是{:.2f}.".format(r2_score_lasso))

#弹性网
from sklearn.linear_model import ElasticNet
enet=ElasticNet(alpha=alpha,l1_ratio=0.7)
y_pred_enet=enet.fit(X_train,y_train).predict(X_test)
r2_score_enet=r2_score(y_test,y_pred_enet)
print("enet")
print("计算出来的r2值是{:.2f}".format(r2_score_enet))
plt.plot(enet.coef_,color="lightgreen",linewidth=2,label="elastic net coefficients")
plt.plot(lasso.coef_,color="gold",linewidth=2,label="lasso coefficients")
plt.plot(coef,'--',color="navy",label="original coefficients")
plt.legend(loc="best")
plt.title("lasso r2 的值是{:.2f}.elastic 的r2值是{:.2f}.".format(r2_score_lasso,r2_score_enet))
plt.show()
