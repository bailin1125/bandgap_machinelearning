#这是一个matminer的测试性项目，用于比较不同数据库数据

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
import pandas as pd
import numpy as np

#首先设置pandas的显示设置，保证都能显示出来
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

c=CitrineDataRetrieval(api_key='QgqmY9PPathNDgu6gJjnTQtt')
df=c.get_dataframe(criteria={'data_type':'EXPERIMENTAL','max_results':100},properties=['Band gap','Temperature'],common_fields=['chemicalFormula'])
df.to_csv('duibi.csv')
df.rename(columns={'Band gap':'Experimnetal band gap'},inplace=True)
df.head()

#然后针对每种组成，从mp数据库中计算的带隙，找到对应的最稳定结构的值
from pymatgen import MPRester,Composition
mpr=MPRester()
def get_mp_bandgap(formula):
    #这个函数的作用是给定一定的化学组成，返回稳定状态的带隙
    #而mo数据库需要用到interger的化学式
    reduced_formula=Composition(formula).get_integer_formula_and_factor()[0]
    struct_list=mpr.get_data(reduced_formula)
    if struct_list:
        return sorted(struct_list,key=lambda e:e['energy_per_atom'])[0]['band_gap']
df['Computed band gap']=df['chemicalFormula'].apply(get_mp_bandgap)

from matminer.figrecipes.plot import PlotlyFig

pf=PlotlyFig(df,x_title='Experimental band gap (ev)',y_title='Computed band gap (ev)',mode='notebook',fontsize=20,ticksize=15)
pf.xy([('Experimental band gap','Computed band gap'),([0,10],[0,10])],modes=['markers','lines'],lines=[{},{'color':'black','dash':'dash'}],labels='chemicalFormula',showlegends=False)
df.head()