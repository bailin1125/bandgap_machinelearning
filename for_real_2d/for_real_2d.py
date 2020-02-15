#这个程序的目的是，通过添加bulkgap和is——daoti的特征，来预测二维材料的带隙
from matminer.figrecipes.plot import PlotlyFig
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import  cross_val_score,KFold,cross_val_predict,GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def pre_gap_forjiyuan():
    #首先载入我们之前建立的vector
    data=pd.read_csv('2d_bulk.csv')
    print(data.columns)
    print(data.describe())
    df=pd.read_csv('vector_new_plustitle.csv')
    print(df.columns)
    print(df.describe())
    df['is_daoti']=np.nan
    df['bulk_gap']=np.nan

    j=0

    print("首先我们需要把之前bulk的is_daoti和预测的gaps放到vector中")
    print("有个问题，某些材料id没有对应信息，只能采取填充法,先规定nan，然后用平均值填充")

    print(len(df.index))
    
    for i in range(len(df.index)):
        #print("this is {}th".format(i+1))
        for j in range(len(data.index)):
            str=data.ix[j,'mp_id']
            str=str[3:]              
            if(eval(str)==df.ix[i,'id']):
            #print(df.ix[j,'id'])
            #print(str)
                df.ix[i,'is_daoti']=data.ix[j,'is_daoti']
                df.ix[i,'bulk_gap']=data.ix[j,'gaps']
                break  
    df=df.fillna(method="ffill")
    #df.to_csv('plus_bulk_isdaoti_2dvector.csv')
    print(df[df.isnull().values==True])

    y_gap=df['gap'].values
    y_m=df['efm'].values

    unwanted=['gap','id','efm']
    X_df=df.drop(unwanted,axis=1,inplace=False)
    X_gap=X_df.values
    
    X_gap=preprocessing.scale(X_gap)
    X_m=X_gap
    crossvalidation=KFold(n_splits=5,shuffle=True,random_state=2)



    #首先进行线性回归
    print("首先进行线性回归")
    #print(metrics.SCORERS.keys())
    lr=LinearRegression()    
    lr.fit(X_gap,y_gap)
    #看一下我们的结果如何
    print("线性训练的r2是:{}".format(round(lr.score(X_gap,y_gap),3)))
    print("训练后的RMSE是{:.3f}".format(np.sqrt(mean_squared_error(y_true=y_gap,y_pred=lr.predict(X_gap)))))
    scores=cross_val_score(lr,X_gap,y_gap,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)   
    rmse_scores=[]    
    for s in scores:
        rmse_scores.append(np.sqrt(abs(s)))     
    print(rmse_scores)
    r2_scores=cross_val_score(lr,X_gap,y_gap,scoring='r2',cv=crossvalidation,n_jobs=1)
    print("相关系数是：{}".format(r2_scores))
    print("现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(np.abs(r2_scores))))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores))))


    #然后开始进行随机森林的预测
    print("开始进行随机森林的预测")
    rf=RandomForestRegressor(n_estimators=90,max_features=10,max_depth=12,min_samples_split=2,random_state=1)  
    rf.fit(X_gap,y_gap)
    print("随机森林的r2是:{}".format(round(rf.score(X_gap,y_gap),3)))
    print("随机森林的是RMSE是:{}".format(round(np.sqrt(mean_squared_error(y_true=y_gap,y_pred=rf.predict(X_gap))),3)))
    
    print("看一下回归效果和什么关系更加密切")
    importances=rf.feature_importances_  
    included=X_df.columns.values
    indices=np.argsort(importances)[::-1]
    pf=PlotlyFig(y_title='importance(%)',title='Feature by importances gap',fontsize=20,ticksize=15)
    pf.bar(x=included[indices][0:10],y=importances[indices][0:10])


    scores=cross_val_score(rf,X_gap,y_gap,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    r2_scores=cross_val_score(rf,X_gap,y_gap,scoring='r2',cv=crossvalidation,n_jobs=1)
    rmse_scores_rf=np.sqrt(abs(scores))
    print("r2 分别是：{}".format(r2_scores))
    
    print("对于随机森林，现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(r2_scores)))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores_rf))))
    print("请看随机森林的结果展示")
    pf_rf = PlotlyFig(x_title='2d HSE calculate gap(ev)',y_title='Random forest predicated 2d gap(ev)',title='Random forest regression',filename='rf_regression.html')
   
   #这里可以用rf.predict(X)来代替交叉验证误差的预测项
    pf_rf.xy([(y_gap, cross_val_predict(rf, X_gap, y_gap, cv=crossvalidation)), ([0, 100], [0, 100])], 
      labels=[], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)



    #这一部分是调参的过程

    #print("首先是n_estimaters的个数和max_features进行调参")
    #param_test1={'n_estimators':range(50,130,10),'max_features':range(5,15)}
    #gsearch1=gridsearchcv(estimator=randomforestregressor(random_state=10),
    #                     param_grid=param_test1,scoring='neg_mean_squared_error',cv=5)
    #gsearch1.fit(x_gap,y_gap)
    #print(gsearch1.cv_results_)
    #print("最好的参数是{}".format(gsearch1.best_params_))
    #print("最好的均方误差是{}".format(gsearch1.best_score_))

    #print("得到了最好的n_estimators是90，最大特征数是10")
    

    #print("最后是对于max_depth和min_samples_split的调参")
    #param_test3={'max_depth':range(4,20,2),'min_samples_split':range(2,5,1)}
    #gsearch3=gridsearchcv(estimator=randomforestregressor(random_state=10,n_estimators=90,max_features=10),
    #                      param_grid=param_test3,scoring='neg_mean_squared_error',cv=5)
    #gsearch3.fit(x_gap,y_gap)
    #print(gsearch3.cv_results_)
    #print("最好的参数是{}".format(gsearch3.best_params_))
    #print("最好的准确率是{}".format(gsearch3.best_score_))

    #print("得到了最好的max_depth是12，min_samples_split是2")



    #开始进行有效质量的预测
    print("开始进行有效质量预测")
    print("首先，支持向量机")
    svm_m=svm.SVR(gamma='scale',C=1.0)
    svm_m.fit(X=X_m,y=y_m)
    print("支持向量机的r2是{:.3f}".format(svm_m.score(X_m,y_m)))
    print("支持向量机的是RMSE是:{}".format(round(np.sqrt(mean_squared_error(y_true=y_m,y_pred=svm_m.predict(X_m))),3)))
    scores=cross_val_score(svm_m,X_m,y_m,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    r2_scores=cross_val_score(svm_m,X_m,y_m,scoring='r2',cv=crossvalidation,n_jobs=1)
    rmse_scores_rf=np.sqrt(abs(scores))
    print("r2 分别是：{}".format(r2_scores))
    
    print("对于支持向量机，现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(r2_scores)))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores_rf))))


    print("没错，开始进行随机森林的预测")
    rf_m=RandomForestRegressor(n_estimators=120,random_state=1)  
    rf_m.fit(X_m,y_m)
    print("随机森林的r2是:{}".format(round(rf_m.score(X_m,y_m),3)))
    print("随机森林的是RMSE是:{}".format(round(np.sqrt(mean_squared_error(y_true=y_m,y_pred=rf_m.predict(X_m))),3)))
    
    print("看一下预测有效质量效果和什么关系更加密切")
    importances=rf_m.feature_importances_  
    included=X_df.columns.values
    indices=np.argsort(importances)[::-1]
    pf=PlotlyFig(y_title='importance(%)',title='Feature by importances efm',fontsize=20,ticksize=15)
    pf.bar(x=included[indices][0:10],y=importances[indices][0:10])


    scores=cross_val_score(rf_m,X_m,y_m,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    r2_scores=cross_val_score(rf_m,X_m,y_m,scoring='r2',cv=crossvalidation,n_jobs=1)
    rmse_scores_rf=np.sqrt(abs(scores))
    print("r2 分别是：{}".format(r2_scores))
    
    print("对于随机森林，现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(r2_scores)))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores_rf))))
    print("请看随机森林的结果展示")
    pf_rf = PlotlyFig(x_title='2d efm calculate efm(ev)',y_title='Random forest predicated 2d efm',title='Random forest regression',filename='rf_regression.html')
   
   #这里可以用rf.predict(X)来代替交叉验证误差的预测项
    pf_rf.xy([(y_m, cross_val_predict(rf_m, X_m, y_m, cv=crossvalidation)), ([0, 100], [0, 100])], 
      labels=[], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)
    print("all work done!")












def main():
    print("首先我们来针对第一基元，预测一下gap和efm")
    pre_gap_forjiyuan()
    #print("接着看一下不同分割训练集和测试集上的效果\n")
    #train_split()

if __name__ == '__main__':
    main()
   