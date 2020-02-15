#这个程序我们来预测体材料的带隙
from matminer.datasets.convenience_loaders import load_elastic_tensor
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.conversions import BaseFeaturizer

from matminer.featurizers.composition import OxidationStates
from matminer.featurizers.composition import ElementProperty

from matminer.featurizers.structure import AverageBondAngle
from matminer.featurizers.structure import AverageBondLength


from matminer.featurizers.structure import DensityFeatures

from matminer.featurizers.bandstructure import BandFeaturizer
from matminer.featurizers.dos import  DOSFeaturizer
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

def generate_data(name):
    #这个函数作用，输入是指定的文件名，输出增加了gaps，is_daoti，以及其他共计145特征的完整向量矩阵
    #name='test_plus_gaps.csv'
    df=pd.read_csv(name,index_col=[0])
    df['gaps']=-10.0   
    df_gap=pd.read_csv("gaps.csv",index_col = [0])
    print(df_gap.index)
    i=0    
    str_s=""
    for j in range(len(df_gap.index)):
        #先打印二者的id
       # print(df.index[i])
        str_s='mp-'+str(df_gap.index[j])
        if(str_s==df.index[i]):
            df.iloc[i,-1]=df_gap.iloc[j,0]
            i=i+1
            #print("确实一样") 
    print("合并完毕")

    #同样的方法我们来建立不同的分类
    df['is_daoti']=-2
    for i in range(len(df.index)):
        if(df.ix[i,-2]==0):
            df.ix[i,-1]=1
        else:
            df.ix[i,-1]=0
    print("分类feature建立完成")   
    
#首先使用describe获得对于数据的整体把握
    print(df.describe())
    df.describe().to_csv('general_look_jie.csv')
#通过观察数据发现并没有什么异常之处
    df=StrToComposition().featurize_dataframe(df,'full_formula',ignore_errors=True)
    print(df.head())   
    #print(df['composition'])
    ep_feat=ElementProperty.from_preset(preset_name='magpie')
    df=ep_feat.featurize_dataframe(df,col_id='composition',ignore_errors=True)#将composition这一列作为特征化的输入
    print(df.head())
    #print(ep_feat.citations())
    #df.to_csv("plus the composition.csv")
    #以上这部分是将formula转化为composition并转化feature

    df=CompositionToOxidComposition().featurize_dataframe(df,col_id='composition')#引入了氧化态的相关特征
    os_feat=OxidationStates()
    df=os_feat.featurize_dataframe(df,col_id='composition_oxid')
    new_name='2d_vector_plus.csv'
    df.to_csv(new_name)

#def generate_formula():
     
    

def random_class():
    X_df=pd.DataFrame()
   
    df=pd.read_csv('full_vector_all.csv',index_col = [0])
    df=df.fillna(df.mean())
    print(df.index.name) 
    #print(df.index)    
    print(df[df.isnull().values==True])        
    df.index.name='mp_id'
    print(df.index.name)
    #读取数据，补充缺失值，更改index
   
    y=df['is_daoti'].values
    uwnanted_columns=['band_gap.optimize_structure_gap','gaps','full_formula','composition','composition_oxid','is_daoti']
    X_df=df.drop(uwnanted_columns,axis=1,inplace=False)
    print(X_df[X_df.isnull().values==True]) 
    X=X_df.values
    
    #选定相关的feature，把原始的区分画出来
    X_fig=df.loc[:,['maximum MendeleevNumber','range oxidation state']]
    X_fig=X_fig.values
    print(X_fig.shape)    
    plt.scatter(X_fig[y==0,0],X_fig[y==0,1],color='red')
    plt.scatter(X_fig[y==1,0],X_fig[y==1,1],color='blue')
    plt.xlabel('maximum MendeleevNumber')
    plt.ylabel('range oxidation state')
    plt.title('is_daoti classifier (red means no daoti,blue means daoti)')
    plt.show()

    
    crossvalidation=KFold(n_splits=10,shuffle=True,random_state=10)
    X=preprocessing.scale(X)
    #print(np.mean(X))

    
    #print("首先是支持向量机")
    #svm_=svm.SVC(kernel='rbf',random_state=1,class_weight='balanced')
    #print("先看一下默认的结果：")
    #svm_.fit(X,y)
    #y_pre_svm=svm_.predict(X)
    #print("准确度是{}".format(metrics.accuracy_score(y,y_pre_svm)))
    #print("召回率是{}".format(metrics.recall_score(y,y_pre_svm)))
    #print("试着先进行调参")

   
    #a=np.array(range(100,200,10))
    #c=list(1./a)
    #print(c)
    #param_test_svm={'gamma':c,'C':[0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8]}
    #gsearch_svm=GridSearchCV(estimator=svm.SVC(random_state=10),
    #                      param_grid=param_test_svm,scoring='accuracy',cv=5)
    #gsearch_svm.fit(X,y)
    #print(gsearch_svm.cv_results_)
    #print("最好的参数是{}".format(gsearch_svm.best_params_))
    #print("最好的准确率是{}".format(gsearch_svm.best_score_))
    
   
    #svm_=svm.SVC(kernel='rbf',C=1.2,gamma=0.00625,random_state=1,class_weight='balanced')
    #scores_svm=cross_val_score(svm_,X,y,scoring='accuracy',cv=crossvalidation,n_jobs=1)
    #print(scores_svm)      
    #print("现在展示交叉验证的结果")    
    #print("经过{}次交叉验证，准确度的平均值是{:.3f}".format(len(scores_svm),np.mean(scores_svm)))

    #首先是随机森林的分类方法
    print("然后是随机森林")
    clf=RandomForestClassifier(n_estimators=100,oob_score=False,random_state=2)
    clf.fit(X,y)
    y_pre=clf.predict(X)
    print("看一下y:{}".format(y))
    print("看一下y_pre:{}".format(y_pre))
    print("对于全部样本训练的准确度是:{}".format(round(clf.score(X,y),3)))
    print("全部样本，准确率是{}".format(metrics.accuracy_score(y,y_pre)))
    print("全部样本，召回率是{}".format(metrics.recall_score(y,y_pre)))
    print("全部样本，精度是{}".format(metrics.precision_score(y,y_pre)))

     
    #但是需要注意的是，我们还需要进行交叉检验
    crossvalidation=KFold(n_splits=10,shuffle=True,random_state=10)
    scores=cross_val_score(clf,X,y,scoring='accuracy',cv=crossvalidation,n_jobs=1)
    print(scores)    
    print("现在展示交叉验证的结果")    
    print("经过{}次交叉验证，准确性的平均值是{:.3f}".format(len(scores),np.mean(scores)))

    #展示feature的重要程度
    importances=clf.feature_importances_    
    included=X_df.columns.values
    indices=np.argsort(importances)[::-1]
    pf=PlotlyFig(y_title='importance(%)',title='Feature by importances(classfier)',fontsize=20,ticksize=15)
    pf.bar(x=included[indices][0:10],y=importances[indices][0:10])

    print("最后，用最好的参数进行验证效果")     
    clf=RandomForestClassifier(n_estimators=100,max_features=14,max_depth=16,min_samples_split=6,random_state=3)
    scores_best=cross_val_score(clf,X,y,scoring='accuracy',cv=crossvalidation,n_jobs=1)
    print(scores)    
    print("现在展示交叉验证的结果")   
    print("经过{}次交叉验证，准确性的平均值是{:.3f}".format(len(scores),np.mean(scores)))

   
    

    #对随机森林进行调参

    #print("首先是n_estimaters的个数进行调参")
    #param_test1={'n_estimators':range(50,110,10)}
    #gsearch1=GridSearchCV(estimator=RandomForestClassifier(random_state=10),
    #                      param_grid=param_test1,scoring='accuracy',cv=5)
    #gsearch1.fit(X,y)
    #print(gsearch1.cv_results_)
    #print("最好的参数是{}".format(gsearch1.best_params_))
    #print("最好的准确率是{}".format(gsearch1.best_score_))

    #print("其次是对于max_features的调参")
    #param_test2={'max_features':range(6,15)}
    #gsearch2=GridSearchCV(estimator=RandomForestClassifier(random_state=10,n_estimators=80),
    #                      param_grid=param_test2,scoring='accuracy',cv=5)
    #gsearch2.fit(X,y)
    #print(gsearch2.cv_results_)
    #print("最好的参数是{}".format(gsearch2.best_params_))
    #print("最好的准确率是{}".format(gsearch2.best_score_))

    #print("最后是对于max_depth和min_samples_split的调参")
    #param_test3={'max_depth':range(8,20,2),'min_samples_split':range(2,8,2)}
    #gsearch3=GridSearchCV(estimator=RandomForestClassifier(random_state=10,n_estimators=80,max_features=14),
    #                      param_grid=param_test3,scoring='accuracy',cv=5)
    #gsearch3.fit(X,y)
    #print(gsearch3.cv_results_)
    #print("最好的参数是{}".format(gsearch3.best_params_))
    #print("最好的准确率是{}".format(gsearch3.best_score_))
    
    
    
    


    ##然后是朴素贝叶斯
    #print("然后是朴素贝叶斯")
    #bayes=GaussianNB()
    #bayes.fit(X,y)
    #print("训练的准确度是:{}".format(round(bayes.score(X,y),3)))
    ##print("训练后的RMSE是{:.3f}".format(np.sqrt(mean_squared_error(y_true=y,y_pred=bayes.predict(X)))))    
    #scores_bayes=cross_val_score(bayes,X,y,scoring='accuracy',cv=crossvalidation,n_jobs=1)
    #print(scores_bayes)  
    #r2_scores_bayes=cross_val_score(clf,X,y,scoring='r2',cv=crossvalidation,n_jobs=1)
    #print(r2_scores_bayes)
    #print("现在展示交叉验证的结果")
    #print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores_bayes),np.mean(np.abs(r2_scores_bayes))))
    #print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores_bayes),np.mean(scores_bayes)))

    #然后是支持向量机
    #然后正式预测体材料的分类 

    clf.fit(X,y)  
    X_2d=pd.DataFrame()
    df_2d=pd.read_csv('2d_vector_plus.csv',index_col = [0])    
    print(df_2d.index.name)       
    print(df_2d[df_2d.isnull().values==True])        
    df_2d.index.name='mp_id'
    print(df_2d.index.name)
    #读取数据，补充缺失值，更改index   
    uwnanted_columns=['band_gap.optimize_structure_gap','gaps','full_formula','composition','composition_oxid','is_daoti']
    X_2d=df_2d.drop(uwnanted_columns,axis=1,inplace=False)
    X_2d=X_2d.fillna(X_2d.mean())
    X_2d_value=X_2d.values
    X_2d_value=preprocessing.scale(X_2d_value)
    y_pre_2d=clf.predict(X_2d_value)
    df_2d['is_daoti']=2
    for i in range(len(df_2d['is_daoti'].values)):
        df_2d.ix[i,['is_daoti']]=y_pre_2d[i]
    
    df_2d.to_csv('预测是否为导体.csv')

    

    print("all work done")
def pre():

    ##导入之后首先进行测试是否有缺失值，并且命名行索引，以及添加gaps数值
    X=pd.DataFrame()
    df=pd.read_csv('full_vector_all.csv',index_col = [0])     
    print(df.index)     
    print(df.index.name)    
    df.index.name='mp_id'
    print(df.index.name)
   
    y=df['gaps'].values
    uwnanted_columns=['band_gap.optimize_structure_gap','gaps','full_formula','composition','composition_oxid']
    X=df.drop(uwnanted_columns,axis=1,inplace=False)
    X=X.fillna(X.mean())
    print(X[X.isnull().values==True])   

    #首先我想先规定下数据的大小范围，对于范围超额的进行近似
    #for co in df.columns:
    #    for row in df.index:            
    #        if(df.loc[row,co]<0.000001 and df.loc[row,co]>-0.000001 and df.loc[row,co]!=0):
    #            print(df.loc[row,co])
    #            df.loc[row,co]=0
    #        elif(df.loc[row,co]>1000000):
    #            print(df.loc[row,co])
    #            df.loc[row,co]=1000000  
     
    #X.to_csv("delete_af_vector.csv")           
    print("这里我们规定hse作为预测项，然后就是其他所有的数字项都是features")    
    print("现在有{}个可能的特征\n\n".format(X.shape[1]))
    print("X 的维度是{}".format(X.shape))   
    print(X[X.isnull().values==True])   
    print(X.isnull().values.any())
    X_pr = X.values
    X_pr=preprocessing.scale(X_pr)


    #首先是线性回归
    lr=LinearRegression()    
    #print(X)
    lr.fit(X_pr,y)   
    print("训练的r2是:{}".format(round(lr.score(X_pr,y),3)))
    print("训练后的RMSE是{:.3f}".format(np.sqrt(mean_squared_error(y_true=y,y_pred=lr.predict(X_pr)))))

    #但是需要注意的是，我们还需要进行交叉检验
    crossvalidation=KFold(n_splits=10,shuffle=True,random_state=1)
    scores=cross_val_score(lr,X_pr,y,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    print(scores)
   # print("暂停30s休息看看结果")
    #time.sleep(30)
    rmse_scores=[]
    
    for s in scores:
        #print(s)
        #print(np.sqrt(abs(s)))
        rmse_scores.append(np.sqrt(abs(s)))        

    print(rmse_scores)
    r2_scores=cross_val_score(lr,X_pr,y,scoring='r2',cv=crossvalidation,n_jobs=1)
    print(r2_scores)

    print("现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(np.abs(r2_scores))))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores))))

    #到这里我们发现预测的结果是不错的
    #但是我们还是需要画图看一下效果

    pf=PlotlyFig(x_title='HSE calculate gap(ev)',y_title='Predicated gap(ev)',title='Linear regression',filename='lr_regression.jpg')
    pf.xy(xy_pairs=[(y,cross_val_predict(lr,X_pr,y,cv=crossvalidation)),([0,400],[0,400])],labels=df['full_formula'],modes=['markers','lines'],lines=[{},{'color':'black','dash':'dash'}],showlegends=False)
    print("这就是线性回归的威力，感觉还是不错的")
    print("\n\n")

    #现在我们尝试使用随机森林来看一下结果如何
    rf=RandomForestRegressor(n_estimators=100,random_state=1)
    rf.fit(X_pr,y)
    print("随机森林的r2是:{}".format(round(rf.score(X_pr,y),3)))
    print("随机森林的是RMSE是:{}".format(round(np.sqrt(mean_squared_error(y_true=y,y_pred=rf.predict(X_pr))),3)))
    #单看整个数据集上效果还是不错的

    importances=rf.feature_importances_
    #print(importances)
    included=X.columns.values
    indices=np.argsort(importances)[::-1]
    #print(indices)

    pf=PlotlyFig(y_title='importance(%)',title='Feature by importances',fontsize=20,ticksize=15)
    pf.bar(x=included[indices][0:10],y=importances[indices][0:10])


    scores=cross_val_score(rf,X_pr,y,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    r2_scores=cross_val_score(rf,X_pr,y,scoring='r2',cv=crossvalidation,n_jobs=1)
    rmse_scores_rf=np.sqrt(abs(scores))
    
    print("对于随机森林，现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(np.abs(r2_scores))))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores_rf))))
    print("请看随机森林的结果展示")
    pf_rf = PlotlyFig(x_title='HSE calculate gap(ev)',y_title='Random forest predicated gap(ev)',title='Random forest regression',filename='rf_regression.html')
   
   #这里可以用rf.predict(X)来代替交叉验证误差的预测项
    pf_rf.xy([(y, cross_val_predict(rf, X_pr, y, cv=crossvalidation)), ([0, 450], [0, 450])], 
      labels=df['full_formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)



    #现在开始正式进行测试了
    rf.fit(X_pr,y)
    print("正式开始进行预测")
    df_2d=pd.read_csv('预测是否为导体.csv',index_col = [0])  
    uwnanted_columns=['band_gap.optimize_structure_gap','gaps','full_formula','composition','composition_oxid']
    X_2d=df_2d.drop(uwnanted_columns,axis=1,inplace=False)
    X_2d=X_2d.fillna(X_2d.mean())   
    print(X_2d[X_2d.isnull().values==True]) 
    
    df_2d.index.name='mp_id'
    print(df_2d.index.name)
    #读取数据，补充缺失值，更改index   
    
    
   
    X_2d_value=X_2d.values
    X_2d_value=preprocessing.scale(X_2d_value)
    y_pre_2d=rf.predict(X_2d_value)
    df_2d['gaps']=-10
    for i in range(len(df_2d['gaps'].values)):
        df_2d.ix[i,['gaps']]=y_pre_2d[i]
    
    df_2d.to_csv('2d_bulk.csv')

    

    print("all work done")



    print("\n\n") 
    
def train_split():

    name='full_vector_all.csv'
    df=pd.read_csv(name,index_col = [0])   
    
    y=df['is_daoti'].values
    uwnanted_columns=['band_gap.optimize_structure_gap','gaps','full_formula','composition','composition_oxid','is_daoti']
    X_df=df.drop(uwnanted_columns,axis=1,inplace=False)
    print(X_df[X_df.isnull().values==True]) 
    X=X_df.values
      
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    print("\n")
    print("我们看下这些分开的不同测试集合训练集")   
    
    rf_reg=RandomForestClassifier(n_estimators=100,random_state=1)
    rf_reg.fit(X_train,y_train)
    #下面我们看一下进行训练集、测试集划分之后的效果
    print("训练集的准确度是：{:.3f}".format(rf_reg.score(X_train,y_train)))
    print("训练集的rmse是:{:.3f}".format(np.sqrt(mean_squared_error(y_true=y_train,y_pred=rf_reg.predict(X_train)))))
    print("测试集上的准确度是:{:.3f}".format(rf_reg.score(X_test,y_test)))
    print("测试集上的rmse是:{:.3f}".format(np.sqrt(mean_squared_error(y_true=y_test,y_pred=rf_reg.predict(X_test)))))

    #然后我们还是来画图
    pf_rf=PlotlyFig(x_title='Bulk modulus prediction residual (Gpa)',y_title='Probability',title='Random forest regression residuals',filename='rf_regression_residuals.html')
    hist_plot=pf_rf.histogram(data=[y_train-rf_reg.predict(X_train),y_test-rf_reg.predict(X_test)],histnorm='probability',colors=['blue','red'],return_plot=True)
    hist_plot['data'][0]['name']='train'
    hist_plot['data'][1]['name']='test'
    pf_rf.create_plot(hist_plot)

    print('\n\n')
    #现在我们需要看看哪些feature是最重要的

def test_files():
    df=load_elastic_tensor()
    df.to_csv('原始elastic数据.csv')
    print(df.columns)

    unwanted_columns=['volume','nsites','compliance_tensor','elastic_tensor','elastic_tensor_original','K_Voigt','G_Voigt','K_Reuss','G_Reuss']
    df=df.drop(unwanted_columns,axis=1)
    print(df.head())
    df.to_csv('扔掉不需要的部分.csv')
def main():
    #print("先测试人家的怎么跑出来的")
    #test_files()
    #print("首先我们先来看一下如何生成有效的数据")
    #generate_data('2d_data_get.csv')
    #print("我们开始进行分类")
    #random_class()
    print("分完类了，接着干吧，生成好的数据进行机器学习及预测")
    pre()
    #print("接着看一下不同分割训练集和测试集上的效果\n")
    #train_split()

if __name__ == '__main__':
    main()
   
