print("now let's do the regression model")
#这个是简单的线性回归的项目
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
#首先开始载入数据
diabets=datasets.load_diabetes()

#仅使用一个特征
diabets_X=diabets.data[:,np.newaxis,2]

#将数据分割放到训练集和测试集
diabets_X_train=diabets_X[:-20]
diabets_X_test=diabets_X[-20:]

#将目标值放到测试集和训练集
diabets_y_train=diabets.target[:-20]
diabets_y_test=diabets.target[-20:]
#创造线性回归对象
regr=linear_model.LinearRegression()

#使用训练集训练模型
regr.fit(diabets_X_train,diabets_y_train)

#进行预测
diabets_y_pred=regr.predict(diabets_X_test)

#显示相关系数和均方误差
mean_squ=mean_squared_error(diabets_y_test,diabets_y_pred)
print('相关系数是：{}\n'.format(regr.coef_))
print("均方误差是:{}\n",format(mean_squ))
print("方差是:{:.2f}.".format(r2_score(diabets_y_test,diabets_y_pred)))

#然后我们来打印相关的结果（绘图）
plt.scatter(diabets_X_test,diabets_y_test,color="black")
plt.plot(diabets_X_test,diabets_y_pred,color="blue",linewidth=3)
#this is the xaxis and yaxis set location
plt.xticks(())

plt.yticks(())
plt.show()