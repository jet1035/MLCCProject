#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 22:34:17 2025

@author: joshtorres
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve


seed = 5353456

#%%

def getMaster():

    #% datasets & features

    # median individual income dataset
        # zipcode
        # median individual income
    # population dataset
        # population
    # heat vulnerability index dataset
        # hvi
    # greenstreets dataset
        # number of greenstreets (maybe consider dropping?)
        # total acres of greenstreets
    # ll84 dataset (only data on buildings that are 50000+ sq ft)
        # mean energy star score
        # mean gap between target and actual energy star scores
        # electricity use generated from onsite renewable systems (kWh)
        # green power onsite (kWh)
        # weather normalized site natural gas use (therms)
    # pluto dataset
        # most common land use 
        # total units
        # total building areas (sqft)
        # borough
    # community gardens dataset
        # number of community gardens
    # trees
        # number of trees
        # average tree health score
        # neighborhood
        
    #% median income

    median_income_data = pd.read_csv('median_individual_income_2023.csv')
    median_income_data.rename(columns={'Entity properties name': 'zipcode', 'Variable observation value': 'median_income'}, inplace=True)

    master_df = median_income_data[['zipcode','median_income']].copy()

    #% population

    population_data = pd.read_csv('population_2023.csv')
    population_data.rename(columns={'Entity properties name': 'zipcode', 'Variable observation value': 'population'}, inplace=True)
    master_df = pd.merge(master_df, population_data[['zipcode', 'population']], on='zipcode', how='inner')

    #% hvi - outcome

    hvi_data = pd.read_csv('heat_vulnerability_index_rankings_2020.csv')
    hvi_data.rename(columns={'ZIP Code Tabulation Area (ZCTA) 2020': 'zipcode', 'Heat Vulnerability Index (HVI)': 'hvi'}, inplace=True)
    hvi_data_target_cols = hvi_data[['zipcode','hvi']]

    master_df = pd.merge(master_df, hvi_data_target_cols, on='zipcode', how='inner')

    #% greenstreets - this lessens the number of rows in the df by a lot... consider removing

    greenstreets_data = pd.read_csv('greenstreets_2022.csv')

    # remove spaces that are 'archived' or 'retired'
    greenstreets_data = greenstreets_data[greenstreets_data['FEATURESTATUS'] != 'Retired']
    greenstreets_data = greenstreets_data[greenstreets_data['FEATURESTATUS'] != 'Archived']

    # find number of greenstreets by zipcode
    greenstreets_data = greenstreets_data[~greenstreets_data['ZIPCODE'].astype(str).str.contains(',')]
    gs_counts_by_zipcode = greenstreets_data['ZIPCODE'].value_counts().to_dict()
    # make sure zipcodes are strings in both dfs for consistent matching
    master_df['zipcode'] = master_df['zipcode'].astype(str)
    # map the counts to a new column in master_df (fill missing with 0)
    master_df['num_greenstreets'] = master_df['zipcode'].map(gs_counts_by_zipcode).fillna(0).astype(int)

    # find total acres of greenstreets by zipcode
    total_area_gs_by_zipcode = greenstreets_data.groupby('ZIPCODE')['ACRES'].sum().reset_index()
    total_area_gs_by_zipcode.columns = ['zipcode', 'total_acres_greenstreets']
    master_df = master_df.merge(total_area_gs_by_zipcode, on='zipcode', how='left')
    master_df['total_acres_greenstreets'] = (master_df['total_acres_greenstreets'].fillna(0))


    #% ll84

    ll84_data = pd.read_csv('ll84_2022_present.csv')

    # extract just 2022 data (closer to ages of other datasets)
    ll84_data = ll84_data[ll84_data['Calendar Year'] == 2022]
    ll84_data['Postal Code'] = ll84_data['Postal Code'].astype(str)
    ll84_data['Postal Code'] = ll84_data['Postal Code'].str.extract(r'(\d{5})')

    # ENERGY STAR Score
    ll84_data.rename(columns={'ENERGY STAR Score': 'energy_star_score'}, inplace=True)
    ll84_data['energy_star_score'] = pd.to_numeric(ll84_data['energy_star_score'], errors='coerce')
    mean_energy_star_score_by_zipcode = ll84_data.groupby('Postal Code')['energy_star_score'].mean().reset_index()
    mean_energy_star_score_by_zipcode.columns = ['zipcode', 'mean_energy_star_score_50k']
    master_df = master_df.merge(mean_energy_star_score_by_zipcode, on='zipcode', how='left')
    master_df['mean_energy_star_score_50k'] = (master_df['mean_energy_star_score_50k'].fillna(0))

    # Target ENERGY STAR Score in order to get mean gap
    ll84_data.rename(columns={'Target ENERGY STAR Score': 'target_energy_star_score'}, inplace=True)
    ll84_data['target_energy_star_score'] = pd.to_numeric(ll84_data['target_energy_star_score'], errors='coerce')
    ll84_data['gap_energy_star_score'] = ll84_data['target_energy_star_score'] - ll84_data['energy_star_score']
    mean_gap_energy_star_score_by_zipcode = ll84_data.groupby('Postal Code')['gap_energy_star_score'].mean().reset_index()
    mean_gap_energy_star_score_by_zipcode.columns = ['zipcode', 'mean_gap_energy_star_score_50k']
    master_df = master_df.merge(mean_gap_energy_star_score_by_zipcode, on='zipcode', how='left')
    master_df['mean_gap_energy_star_score_50k'] = (master_df['mean_gap_energy_star_score_50k'].fillna(0))

    # Electricity Use – Generated from Onsite Renewable Systems (kWh)
    ll84_data.rename(columns={'Electricity Use – Generated from Onsite Renewable Systems (kWh)': 'electricity_onsite_renew'}, inplace=True)
    ll84_data['electricity_onsite_renew'] = pd.to_numeric(ll84_data['electricity_onsite_renew'], errors='coerce')
    electricity_onsite_by_zipcode = ll84_data.groupby('Postal Code')['electricity_onsite_renew'].sum().reset_index()
    electricity_onsite_by_zipcode.columns = ['zipcode', 'electricity_onsite_renew_50k']
    master_df = master_df.merge(electricity_onsite_by_zipcode, on='zipcode', how='left')
    master_df['electricity_onsite_renew_50k'] = (master_df['electricity_onsite_renew_50k'].fillna(0))

    # Green Power - Onsite (kWh)
    ll84_data.rename(columns={'Green Power - Onsite (kWh)': 'green_power_onsite'}, inplace=True)
    ll84_data['green_power_onsite_50k'] = pd.to_numeric(ll84_data['green_power_onsite'], errors='coerce')
    total_green_power_by_zipcode = ll84_data.groupby('Postal Code')['green_power_onsite'].sum().reset_index()
    total_green_power_by_zipcode.columns = ['zipcode', 'green_power_onsite_50k']
    master_df = master_df.merge(total_green_power_by_zipcode, on='zipcode', how='left')
    master_df['green_power_onsite_50k'] = (master_df['green_power_onsite_50k'].fillna(0))

    #TEMPORARILY REMOVING THIS
    '''
    # Natural Gas – Weather Normalized Site Natural Gas Use (therms)
    ll84_data.rename(columns={'Natural Gas – Weather Normalized Site Natural Gas Use (therms)': 'nat_gas_onsite'}, inplace=True)
    ll84_data['nat_gas_onsite'] = pd.to_numeric(ll84_data['nat_gas_onsite'], errors='coerce')
    total_nat_gas_by_zipcode = ll84_data.groupby('Postal Code')['nat_gas_onsite'].sum().reset_index()
    total_nat_gas_by_zipcode.columns = ['zipcode', 'nat_gas_onsite_50k']
    master_df = master_df.merge(total_nat_gas_by_zipcode, on='zipcode', how='left')
    master_df['nat_gas_onsite_50k'] = (master_df['nat_gas_onsite_50k'].fillna(0))
    '''
    #% pluto

    pluto_data = pd.read_csv('pluto_2020.csv')

    pluto_data['postcode'] = (
        pluto_data['postcode']
        .dropna()                    # Remove NaNs
        .astype(float)              # Make sure it's float
        .astype(int)                # Drop the decimal
        .astype(str)                # Convert to string
        .str.zfill(5)               # Pad with leading zeros if needed
    )

    # landuse stuff
    # goal: have each zipcode be assigned a majority landuse value
    # mostly mostly 01, 02, or 03 = mostly residential; 
    # if mostly 04 and 05, mostly commercial; 
    # if mostly 06, 07, 08, mostly industrial; 
    # if mostly 10 or 11, mostly empty or undeveloped land
    # disregard 09
    # dropping nans, 09s
    pluto_data = pluto_data.dropna(subset=['landuse'])
    pluto_data = pluto_data[pluto_data['landuse'] != '09']
    # group by landuse counts by zipcode
    landuse_counts = pluto_data.groupby('postcode')['landuse'].value_counts().unstack(fill_value=0).reset_index()
    landuse_counts.columns = ['postcode'] + [f"{int(col):02d}" for col in landuse_counts.columns[1:]]
    most_common_landuse = landuse_counts.set_index('postcode').idxmax(axis=1).reset_index()
    most_common_landuse.columns = ['zipcode', 'most_common_landuse_code']
    landuse_map = {
        '01': 1, '02': 1, '03': 1,   # Residential
        '04': 2, '05': 2,           # Commercial
        '06': 3, '07': 3, '08': 3,  # Industrial
        '10': 4, '11': 4            # Undeveloped
    }
    most_common_landuse['most_common_landuse_code'] = most_common_landuse['most_common_landuse_code'].map(landuse_map)
    master_df = master_df.merge(most_common_landuse[['zipcode', 'most_common_landuse_code']], on='zipcode', how='left')
    # just to see
    # Optional: set human-readable labels
    landuse_labels = {
        1: "Residential",
        2: "Commercial",
        3: "Industrial",
        4: "Undeveloped"
    }
    # Count how many ZIP codes fall into each LandUse category
    landuse_counts = master_df['most_common_landuse_code'].value_counts().sort_index()
    # Map numeric labels to readable names for the x-axis
    labels = [landuse_labels.get(code, 'Unknown') for code in landuse_counts.index]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(labels, landuse_counts.values, color='mediumseagreen')
    plt.xlabel("Land Use Category")
    plt.ylabel("Number of ZIP Codes")
    plt.title("Most Common Land Use by ZIP Code")
    plt.tight_layout()
    plt.show()


    # unitstotal
    total_units_by_zipcode = pluto_data.groupby('postcode')['unitstotal'].sum().reset_index()
    total_units_by_zipcode.columns = ['zipcode', 'total_units']
    master_df = master_df.merge(total_units_by_zipcode, on='zipcode', how='left')
    master_df['total_units'] = (master_df['total_units'].fillna(0))

    # bldgarea
    building_areas_by_zipcode = pluto_data.groupby('postcode')['bldgarea'].sum().reset_index()
    building_areas_by_zipcode.columns = ['zipcode', 'total_building_areas']
    master_df = master_df.merge(building_areas_by_zipcode, on='zipcode', how='left')
    master_df['total_building_areas'] = (master_df['total_building_areas'].fillna(0))

    # borough
    borough_mode = (
        pluto_data
          .groupby('postcode')['borough']
          .agg(lambda x: x.mode()[0])  # take the most frequent borough
          .reset_index()
    )
    borough_mode.columns = ['zipcode','borough']
    master_df = master_df.merge(borough_mode, on='zipcode', how='left')


    #% community gardens

    # get block to zipcode dataframe
    block_zipcode_data = pluto_data[['postcode', 'Tax block']]
    block_zipcode_data.columns = ['zipcode', 'block']

    community_garden_data = pd.read_csv('greenthumb_community_gardens_2020.csv')
    community_garden_data = community_garden_data.merge(block_zipcode_data, on='block', how='left')

    # number of community gardens
    community_gardens_by_zipcode = community_garden_data['zipcode'].value_counts().to_dict()
    master_df['num_community_gardens'] = master_df['zipcode'].map(community_gardens_by_zipcode).fillna(0).astype(int)


    #% trees

    # for number of trees and proportion of trees that are healthy
    def getTrees():
        
        df=pd.read_csv('street_tree_census_2015.csv')

        columns=df.columns

        trees=df.iloc[:,[0,6,7,25]]
        tree_data = trees[:]

        condition = (trees['health'].isna() & (trees['status']=='Alive'))
        cleanTrees=trees[~condition]

        cleanTrees=cleanTrees.drop(columns=['status'])

        healthValues={'Good':0,'Fair':1,'Poor':2}
        cleanTrees['health']=cleanTrees['health'].map(healthValues)

        cleanTrees['health'] = cleanTrees['health'].fillna(3).astype(int)
        
        treeCounts=cleanTrees.groupby('postcode').size()

        poorHealth=(cleanTrees['health']==2)

        goodProportion = cleanTrees.groupby('postcode')['health'].apply(lambda x: (x==0).sum()/x.count()).to_frame(name='goodProportion')

        treesZip = pd.concat([treeCounts, goodProportion], axis=1)
        treesZip=treesZip.reset_index()
        treesZip.rename(columns={0:'num_trees','postcode':'zipcode', 'goodProportion': 'proportion_healthy_trees'},inplace=True)
       
        return treesZip

    tree_by_zip = getTrees()


    # Ensure ZIP format matches
    master_df['zipcode'] = master_df['zipcode'].astype(str).str.zfill(5)
    tree_by_zip['zipcode'] = tree_by_zip['zipcode'].astype(str).str.zfill(5)

    # Merge on ZIP code
    master_df = master_df.merge(tree_by_zip, on='zipcode', how='left')
    borough_map={
        'MN':0,
        'SI':1,
        'BX':2,
        'QN':3,
        'BK':4}
    master_df['borough']=master_df['borough'].map(borough_map)
    master_df=master_df.drop(columns='green_power_onsite_50k')
    master_df['zipcode']=master_df['zipcode'].astype(int)
    return master_df


data = getMaster()

#index hvi properly for xgboost
data['hvi'] = data['hvi'] - 1

#%%Train test split

#get X and y first
y = data['hvi'].to_numpy()
X = data.drop(columns = ['hvi','zipcode']).to_numpy()


#Get test data first
#stratify to keep proportion of class labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

#further split to get validation set 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify = y_train, random_state=seed)


#%% Fit standard scaler

#get indices of columns
columns=list(data.columns)
columns.pop(3)

#Columns to scale
scaleCols = [1,2,3,4,5,6,7,9,10,11,12,13,14]

#Fit scaler 
scaler = StandardScaler().fit(X_train[:, scaleCols])

#Copy datasets
X_train_scaled = np.copy(X_train)
X_val_scaled = np.copy(X_val)
X_test_scaled = np.copy(X_test)

#transform
X_train[:, scaleCols] = scaler.transform(X_train[:, scaleCols])
X_val[:, scaleCols] = scaler.transform(X_val[:, scaleCols])
X_test[:, scaleCols] = scaler.transform(X_test[:, scaleCols])

#%%boost
#set hyper parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 2, #learning rate of 2 gets 0.79 average auc, 4 gets 0.75 which do we do????
            'alpha': 10,
            'learning_rate': 0.1,
            'n_estimators':10
        }         
             
#instantiate the classifier 
boost = XGBClassifier(**params).fit(X_train,y_train)

#%get predictions
boost_pred = boost.predict(X_val)
boost_proba = boost.predict_proba(X_val)


#%metrics
boostAccuracy = boost.score(X=X_val, y=y_val)
boostCM = confusion_matrix(y_val, boost_pred)
boostAUC = roc_auc_score(y_val, boost_proba, multi_class='ovr')

#% get metrics for each class
n_classes = boost_proba.shape[1]

#initialize list
boost_class_aucs = []

for i in range(n_classes):
    # create binary labels for class i
    binary_true = (y_val == i).astype(int)  # 1 if true class is i, else 0
    
    # predicted probability for class i
    prob_i = boost_proba[:, i]
    
    # compute AUC for class i
    auc_i = roc_auc_score(binary_true, prob_i)
    boost_class_aucs.append(auc_i)

for i, auc in enumerate(boost_class_aucs):
    print(f"{i}: {auc:.3f}")
    
#%%Neural network
import torch
from torch import nn, optim
from IPython import display

#get tensors for mlp
X_train_pth = torch.as_tensor(X_train, dtype=torch.float)
X_val_pth = torch.as_tensor(X_val, dtype=torch.float)
X_test_pth = torch.as_tensor(X_test, dtype=torch.float)


y_train_pth = torch.as_tensor(y_train).long()
y_val_pth = torch.as_tensor(y_val).long()
y_test_pth = torch.as_tensor(y_test).long()


#remove two rows with nans
mask = ~torch.isnan(X_train_pth).any(dim=1)
X_train_pth = X_train_pth[mask]
y_train_pth = y_train_pth[mask]

#%%
D = 15
C = 5
H = 20

mlp = nn.Sequential(
    nn.Linear(D,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )

#hyper parameters
learning_rate = 0.1
lambda_l2 = 1e-4

#%%Train MLP
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, weight_decay = lambda_l2)

for t in range(1000):
    #Forward Pass
    mlpPred = mlp(X_train_pth) #1
    
    #Compute loss and accuracy
    loss = criterion(mlpPred, y_train_pth)
    score, predicted = torch.max(mlpPred,1)
    acc = (y_train_pth == predicted).sum().item()/len(y_train) #2
    
    print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
    display.clear_output(wait=True)
    
    #3 reset gradients
    optimizer.zero_grad() 
    
    #4
    loss.backward()
    
    #5
    optimizer.step()
    
#%%
out = mlp(X_val_pth)
_, predicted = torch.max(out.data, 1)

print('Accuracy of the network %.4f %%' % (100 * torch.sum(y_val_pth==predicted).double() / len(y_val_pth)))
    

#%%
'''
import torch.nn.functional as F

mlp.eval()
with torch.no_grad():
    mlpPred=mlp(X_val_pth)
    probs = F.softmax(mlpPred, dim=1)

y_true=y_val_pth.cpu().numpy()
y_score=probs.cpu().numpy()

auc = roc_auc_score(y_val, y_score, multi_class='ovr')
print("Final AUC: %.3f" % auc)
'''

#%%
import torch.nn.functional as F


# Compute probabilities as before
mlp.eval()
with torch.no_grad():
    mlpPred = mlp(X_val_pth)
    probs = F.softmax(mlpPred, dim=1)

# Convert to numpy arrays for sklearn
y_true = y_val_pth.cpu().numpy()  # True labels
y_score = probs.cpu().numpy()     # Predicted probabilities

# Get the number of classes
n_classes = y_score.shape[1]

# List to store AUCs
class_aucs = []

# Compute AUC for each class
for i in range(n_classes):
    # Create binary labels for class i (One-vs-Rest)
    binary_true = (y_true == i).astype(int)
    
    # Predicted probability for class i
    prob_i = y_score[:, i]
    
    # Compute AUC for class i
    auc_i = roc_auc_score(binary_true, prob_i)
    class_aucs.append(auc_i)

# Print individual class AUCs
for i, auc in enumerate(class_aucs):
    print(f"Class {i}: AUC = {auc:.3f}")

# Compute and print the overall AUC
overall_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
print(f"Overall AUC: {overall_auc:.3f}")