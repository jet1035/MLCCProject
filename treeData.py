#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:56:32 2025

@author: joshtorres
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%load csv
df=pd.read_csv('trees.csv')

#%%
columns=df.columns

#get tree_id, status, health, post code
trees=df.iloc[:,[0,6,7,25]]

#%%
#check for nans
treeNans=trees.isna().astype(int)

totalNans=treeNans.sum()

'''
Only health has missing values and all of those occur when the tree is dead. Status is redundant and we can just
use health which is better. 

Only ONE row has 

nan(dead)=3, poor=2, fair=1, good=0
'''

#%%look at heatlh
noHealth=trees[trees['health'].isna() & (trees['status']=='Alive')]


#%%drop rows where 'Alive' ~= nan

condition =(trees['health'].isna() & (trees['status']=='Alive'))
cleanTrees=trees[~condition]

cleanTrees=cleanTrees.drop(columns=['status'])

#%%categorize values

healthValues={'Good':0,'Fair':1,'Poor':2}
cleanTrees['health']=cleanTrees['health'].map(healthValues)

#replace nan with 3
cleanTrees['health'] = cleanTrees['health'].fillna(3).astype(int)

#%%
treeCounts=cleanTrees.groupby('postcode').size()

#%%add poor health proportion

poorHealth=(cleanTrees['health']==2)

poorProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==2).sum()/x.count()).to_frame(name='poorProportion')
fairProportion=cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==1).sum()/x.count()).to_frame(name='fairProportion')
goodProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==0).sum()/x.count()).to_frame(name='goodProportion')
deadProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==3).sum()/x.count()).to_frame(name='deadProportion')

#%%
treesZip = pd.concat([treeCounts, deadProportion, poorProportion, fairProportion, goodProportion], axis=1)

#%%
plt.figure()
plt.scatter
plt.show()

#%%
health_props = cleanTrees.groupby('postcode')['health'].value_counts(normalize=True).unstack().fillna(0)

health_props[[0,1,2,3]].plot(
    kind='bar',
    stacked=True,
    color=['green', 'gold', 'orangered', 'grey'],
    figsize=(12,6)
)

plt.xlabel('Zipcode')
plt.ylabel('Proportion')
plt.title('Tree Health Distribution by Zipcode')
plt.legend(['Good', 'Fair', 'Poor', 'Dead'])
plt.tight_layout()
plt.show()

#%%
def getTrees():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    #load csv
    df=pd.read_csv('trees.csv')

    #
    columns=df.columns

    #get tree_id, status, health, post code
    trees=df.iloc[:,[0,6,7,25]]

    #

    '''
    Only health has missing values and all of those occur when the tree is dead. Status is redundant and we can just
    use health which is better. 

    Only ONE row has 

    nan(dead)=3, poor=2, fair=1, good=0
    '''

    #drop rows where 'Alive' ~= nan

    condition =(trees['health'].isna() & (trees['status']=='Alive'))
    cleanTrees=trees[~condition]

    cleanTrees=cleanTrees.drop(columns=['status'])

    #categorize values

    healthValues={'Good':0,'Fair':1,'Poor':2}
    cleanTrees['health']=cleanTrees['health'].map(healthValues)

    #replace nan with 3
    cleanTrees['health'] = cleanTrees['health'].fillna(3).astype(int)

    #
    treeCounts=cleanTrees.groupby('postcode').size()

    #add poor health proportion

    poorHealth=(cleanTrees['health']==2)

    #poorProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==2).sum()/x.count()).to_frame(name='poorProportion')
    #fairProportion=cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==1).sum()/x.count()).to_frame(name='fairProportion')
    goodProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==0).sum()/x.count()).to_frame(name='goodProportion')
    #deadProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==3).sum()/x.count()).to_frame(name='deadProportion')

    #
    treesZip = pd.concat([treeCounts, goodProportion], axis=1)
    #set postcode as a column
    treesZip=treesZip.reset_index()
    treesZip.rename(columns={0:'treeCount','postcode':'zipcode'},inplace=True)

    
    return treesZip

#%%
tree2 = getTrees()



    
                   




