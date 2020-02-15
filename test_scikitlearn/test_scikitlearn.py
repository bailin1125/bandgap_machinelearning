from matminer.datasets.convenience_loaders import load_elastic_tensor
from matminer.featurizers.conversions import StrToComposition 
from matminer.featurizers.composition import ElementProperty
#然后出了element——property之外还有其他的增加特征化算符的方法
#这里简单再引入两种
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates
from matminer.featurizers.structure import DensityFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,cross_val_score
import time
from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def generate_data():
    df=load_elastic_tensor()
    df.to_csv('原始elastic数据.csv')
    print(df.columns)

    unwanted_columns=['volume','nsites','compliance_tensor','elastic_tensor','elastic_tensor_original','K_Voigt','G_Voigt','K_Reuss','G_Reuss']
    df=df.drop(unwanted_columns,axis=1)
    print(df.head())
    df.to_csv('扔掉不需要的部分.csv')


#首先使用describe获得对于数据的整体把握
    print(df.describe())
    df.describe().to_csv('general_look.csv')
#通过观察数据发现并没有什么异常之处
    df=StrToComposition().featurize_dataframe(df,'formula')
    print(df.head())
    df.to_csv('引入composition.csv')

#下一步，我们需要其中一个特征化来增加一系列的特征算符
    ep_feat=ElementProperty.from_preset(preset_name='magpie')
    df=ep_feat.featurize_dataframe(df,col_id='composition')#将composition这一列作为特征化的输入
    print(df.head())
    print(ep_feat.citations())
    df.to_csv('将composition特征化后.csv')

    #开始引入新的特征化算符吧
    df=CompositionToOxidComposition().featurize_dataframe(df,'composition')#引入了氧化态的相关特征
    os_feat=OxidationStates()
    df=os_feat.featurize_dataframe(df,col_id='composition_oxid')
    print(df.head())
    df.to_csv('引入氧化态之后.csv')

    #其实除了基于composition的特征之外还有很多其他的，比如基于结构的
    df_feat=DensityFeatures()
    df=df_feat.featurize_dataframe(df,'structure')
    print(df.head())
    df.to_csv('引入结构中的密度.csv')
    print(df_feat.feature_labels())

def pre():
    print("既然有了这么多数据，我们需要考虑好谁是输入，谁是输出")
    print("这里我们规定K-var作为预测项，然后就是其他所有的数字项都是features")
    df=pd.read_csv('引入结构中的密度.csv')
    print(df.columns)
    y=df['K_VRH'].values
    excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id", 
            "poisson_ratio", "structure", "composition", "composition_oxid"]
    X=df.drop(excluded,axis=1)
    print("现在有{}个可能的特征：\n\n".format(X.shape[1],X.columns.values))
    lr=LinearRegression()
    lr.fit(X,y)
    #看一下我们的结果如何
    print("训练的r2是:{}".format(round(lr.score(X,y),3)))
    print("训练后的RMSE是{:.3f}".format(np.sqrt(mean_squared_error(y_true=y,y_pred=lr.predict(X)))))

    #但是需要注意的是，我们还需要进行交叉检验
    crossvalidation=KFold(n_splits=10,shuffle=False,random_state=1)
    scores=cross_val_score(lr,X,y,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    print(scores)
   # print("暂停30s休息看看结果")
    #time.sleep(30)
    rmse_scores=[]
    
    for s in scores:
        #print(s)
        #print(np.sqrt(abs(s)))
        rmse_scores.append(np.sqrt(abs(s)))        

    print(rmse_scores)
    r2_scores=cross_val_score(lr,X,y,scoring='r2',cv=crossvalidation,n_jobs=1)
    print(r2_scores)

    print("现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(np.abs(r2_scores))))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores))))

    #到这里我们发现预测的结果是不错的
    #但是我们还是需要画图看一下效果
    pf=PlotlyFig(x_title='DFT (MP) bulk modules(Gpa)',y_title='Predicated bulk modules(Gpa)',title='Linear regression',filename='lr_regression.jpg')
    pf.xy(xy_pairs=[(y,cross_val_predict(lr,X,y,cv=crossvalidation)),([0,400],[0,400])],labels=df['formula'],modes=['markers','lines'],lines=[{},{'color':'black','dash':'dash'}],showlegends=False)
    print("这就是线性回归的威力，感觉还是不错的")
    print("\n\n")

    #现在我们尝试使用随机森林来看一下结果如何
    rf=RandomForestRegressor(n_estimators=50,random_state=1)
    rf.fit(X,y)
    print("随机森林的r2是:{}".format(round(rf.score(X,y),3)))
    print("随机森林的是RMSE是:{}".format(round(np.sqrt(mean_squared_error(y_true=y,y_pred=rf.predict(X))),3)))
    #单看整个数据集上效果还是不错的

    importances=rf.feature_importances_
    included=X.columns.values
    indices=np.argsort(importances)[::-1]

    pf=PlotlyFig(y_title='importance(%)',title='Feature by importances',fontsize=20,ticksize=15)
    pf.bar(x=included[indices][0:10],y=importances[indices][0:10])


    scores=cross_val_score(rf,X,y,scoring='neg_mean_squared_error',cv=crossvalidation,n_jobs=1)
    r2_scores=cross_val_score(rf,X,y,scoring='r2',cv=crossvalidation,n_jobs=1)
    rmse_scores_rf=np.sqrt(abs(scores))
    
    print("对于随机森林，现在展示交叉验证的结果")
    print("经过{}次交叉验证，r2的平均值是{:.3f}".format(len(scores),np.mean(np.abs(r2_scores))))
    print("经过{}次交叉验证，rmse的平均值是{:.3f}".format(len(scores),np.mean(np.abs(rmse_scores_rf))))
    print("请看随机森林的结果展示")
    pf_rf = PlotlyFig(x_title='DFT (MP) bulk modulus (GPa)',y_title='Random forest bulk modulus (GPa)',title='Random forest regression',filename='rf_regression.html')
   
   #这里可以用rf.predict(X)来代替交叉验证误差的预测项
    pf_rf.xy([(y, cross_val_predict(rf, X, y, cv=crossvalidation)), ([0, 400], [0, 400])], 
      labels=df['formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)

    print("\n\n") 
    
def train_split():

    df=pd.read_csv('引入结构中的密度.csv')
    y=df['K_VRH'].values
    excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id", 
            "poisson_ratio", "structure", "composition", "composition_oxid"]
    X=df.drop(excluded,axis=1)    
    X['formula']=df['formula']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    print("\n")
    print("我们看下这些分开的不同测试集合训练集")
    print("x训练集是\n{}".format(X_train))
    #time.sleep(30)
    train_formula=X_train['formula']
    X_train=X_train.drop('formula',axis=1)
    test_formula=X_test['formula']
    X_test=X_test.drop('formula',axis=1)
    rf_reg=RandomForestRegressor(n_estimators=100,random_state=1)
    rf_reg.fit(X_train,y_train)
    #下面我们看一下进行训练集、测试集划分之后的效果
    print("训练集的r2是：{:.3f}".format(rf_reg.score(X_train,y_train)))
    print("训练集的rmse是:{:.3f}".format(np.sqrt(mean_squared_error(y_true=y_train,y_pred=rf_reg.predict(X_train)))))
    print("测试集上的r2是:{:.3f}".format(rf_reg.score(X_test,y_test)))
    print("测试集上的rmse是:{:.3f}".format(np.sqrt(mean_squared_error(y_true=y_test,y_pred=rf_reg.predict(X_test)))))

    #然后我们还是来画图
    pf_rf=PlotlyFig(x_title='Bulk modulus prediction residual (Gpa)',y_title='Probability',title='Random forest regression residuals',filename='rf_regression_residuals.html')
    hist_plot=pf_rf.histogram(data=[y_train-rf_reg.predict(X_train),y_test-rf_reg.predict(X_test)],histnorm='probability',colors=['blue','red'],return_plot=True)
    hist_plot['data'][0]['name']='train'
    hist_plot['data'][1]['name']='test'
    pf_rf.create_plot(hist_plot)

    print('\n\n')
    #现在我们需要看看哪些feature是最重要的
def test_featurizers():
    df=pd.read_csv('test.csv',index_col=[0])
    df=StrToComposition().featurize_dataframe(df,'formula')
    print(df.head())
#下一步，我们需要其中一个特征化来增加一系列的特征算符
    ep_feat=ElementProperty.from_preset(preset_name='magpie')
    df=ep_feat.featurize_dataframe(df,col_id='composition')#将composition这一列作为特征化的输入
    print(df.head())
    print(ep_feat.citations())
    #df.to_csv('将composition特征化后.csv')

    #开始引入新的特征化算符吧
    df=CompositionToOxidComposition().featurize_dataframe(df,'composition')#引入了氧化态的相关特征
    os_feat=OxidationStates()
    df=os_feat.featurize_dataframe(df,col_id='composition_oxid')
    print(df.head())
    df.to_csv('after_test.csv')


def main():
    #print("首先我们先来看一下如何生成有效的数据")
    #generate_data()
    #print("然后我们用生成好的数据进行机器学习及预测")
    #pre()
    #print("接着看一下不同分割训练集和测试集上的效果\n")
    #train_split()
    print("简单的小测试")
    test_featurizers()

if __name__ == '__main__':
    main()
   
