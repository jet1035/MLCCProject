#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 20:00:00 2025

@author: meganstratton
"""

#%% imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% datasets & features

# FINAL DATAFRAME TO USE FOR MODELS IS AT THE BOTTOM OF THIS FILE CALLED 'DATA'
# below is a summary of the datasets used and the features taken from them

# median individual income dataset
    # zipcode
    # income group (0 if < 50k, 1 if 50k-100k, 2 if 100k-150k, 3 if > 150k)
# population dataset
    # population
# heat vulnerability index dataset (target)
    # hvi
# greenstreets dataset
    # total acres of greenstreets
# ll84 dataset (only data on buildings that are 50000+ sq ft)
    # mean energy star score of a building
    # electricity use generated from onsite renewable systems (kWh) per building
    # green power onsite (kWh) per building
    # weather normalized site natural gas use (therms) per building
# pluto dataset
    # most common land use 
    # residential units per capita
    # average area (sqft) per building
    # borough
# community gardens dataset
    # number of community gardens
# trees dataset
    # number of living trees
    # neighborhood
# college dataset
    # percent with college degree
# justice dataset
    # percent white
    # percent black
    # jail admissions per 100k
# poverty dataset
    # percent poverty
    
#%% high income

median_income_data = pd.read_csv('median_individual_income_2023.csv')
median_income_data.rename(columns={'Entity properties name': 'zipcode', 'Variable observation value': 'median_income'}, inplace=True)
conditions = [
    median_income_data['median_income'] < 50000,
    (median_income_data['median_income'] >= 50000) & (median_income_data['median_income'] < 100000),
    (median_income_data['median_income'] >= 100000) & (median_income_data['median_income'] < 150000),
    median_income_data['median_income'] >= 150000
]
values = [0, 1, 2, 3]
median_income_data['income_group'] = np.select(conditions, values)
master_df = median_income_data[['zipcode','income_group']].copy()
master_df['zipcode'] = master_df['zipcode'].astype(str)

#%% population

population_data = pd.read_csv('population_2023.csv')
population_data.rename(columns={'Entity properties name': 'zipcode', 'Variable observation value': 'population'}, inplace=True)
population_data['zipcode'] = population_data['zipcode'].astype(str)
master_df = pd.merge(master_df, population_data[['zipcode', 'population']], on='zipcode', how='inner')

#%% hvi - outcome

hvi_data = pd.read_csv('heat_vulnerability_index_rankings_2020.csv')
hvi_data.rename(columns={'ZIP Code Tabulation Area (ZCTA) 2020': 'zipcode', 'Heat Vulnerability Index (HVI)': 'hvi'}, inplace=True)
hvi_data_target_cols = hvi_data[['zipcode','hvi']]
hvi_data_target_cols['zipcode'] = hvi_data_target_cols['zipcode'].astype(str)

master_df = pd.merge(master_df, hvi_data_target_cols, on='zipcode', how='inner')

#%% greenstreets - this lessens the number of rows in the df by a lot... consider removing

greenstreets_data = pd.read_csv('greenstreets_2022.csv')

# remove spaces that are 'archived' or 'retired'
greenstreets_data = greenstreets_data[greenstreets_data['FEATURESTATUS'] != 'Retired']
greenstreets_data = greenstreets_data[greenstreets_data['FEATURESTATUS'] != 'Archived']

# find total acres of greenstreets by zipcode
total_acres_gs_by_zipcode = greenstreets_data.groupby('ZIPCODE')['ACRES'].sum().reset_index()
total_acres_gs_by_zipcode.columns = ['zipcode', 'total_acres_greenstreets']
master_df['zipcode'] = master_df['zipcode'].astype(str)
total_acres_gs_by_zipcode['zipcode'] = total_acres_gs_by_zipcode['zipcode'].astype(str)
master_df = master_df.merge(total_acres_gs_by_zipcode, on='zipcode', how='left')
master_df['total_acres_greenstreets'] = (master_df['total_acres_greenstreets'].fillna(0))


#%% ll84

ll84_data = pd.read_csv('ll84_2022_present.csv')
ll84_data = ll84_data[['Calendar Year', 'Postal Code', 'ENERGY STAR Score', 'Electricity Use – Generated from Onsite Renewable Systems (kWh)', 'Green Power - Onsite (kWh)', 'Natural Gas - Weather Normalized Site Natural Gas Use (therms)']].dropna()

# extract just 2022 data (closer to ages of other datasets)
ll84_data = ll84_data[ll84_data['Calendar Year'] == 2022]
ll84_data['Postal Code'] = ll84_data['Postal Code'].astype(str)
ll84_data['Postal Code'] = ll84_data['Postal Code'].str.extract(r'(\d{5})')

# ENERGY STAR Score
ll84_data.rename(columns={'ENERGY STAR Score': 'energy_star_score'}, inplace=True)
ll84_data['energy_star_score'] = pd.to_numeric(ll84_data['energy_star_score'], errors='coerce')
mean_energy_star_score_by_zipcode = ll84_data.groupby('Postal Code')['energy_star_score'].mean().reset_index()
mean_energy_star_score_by_zipcode.columns = ['zipcode', 'mean_energy_star_score_50k_building']
master_df = master_df.merge(mean_energy_star_score_by_zipcode, on='zipcode', how='left')
# master_df['mean_energy_star_score_50k'] = (master_df['mean_energy_star_score_50k'].fillna(0))

'''
# Target ENERGY STAR Score in order to get mean gap
ll84_data.rename(columns={'Target ENERGY STAR Score': 'target_energy_star_score'}, inplace=True)
ll84_data['target_energy_star_score'] = pd.to_numeric(ll84_data['target_energy_star_score'], errors='coerce')
ll84_data['gap_energy_star_score'] = ll84_data['target_energy_star_score'] - ll84_data['energy_star_score']
mean_gap_energy_star_score_by_zipcode = ll84_data.groupby('Postal Code')['gap_energy_star_score'].mean().reset_index()
mean_gap_energy_star_score_by_zipcode.columns = ['zipcode', 'mean_gap_energy_star_score_50k']
master_df = master_df.merge(mean_gap_energy_star_score_by_zipcode, on='zipcode', how='left')
# master_df['mean_gap_energy_star_score_50k'] = (master_df['mean_gap_energy_star_score_50k'].fillna(0))
'''

# Number of buildings to normalize by
ll84_buildings_by_zipcode = ll84_data['Postal Code'].value_counts().to_dict()
ll84_buildings_df = pd.DataFrame.from_dict(ll84_buildings_by_zipcode, orient='index').reset_index()
ll84_buildings_df.columns = ['zipcode', 'building_count']

# Electricity Use – Generated from Onsite Renewable Systems (kWh)
ll84_data.rename(columns={'Electricity Use – Generated from Onsite Renewable Systems (kWh)': 'electricity_onsite_renew'}, inplace=True)
ll84_data['electricity_onsite_renew'] = pd.to_numeric(ll84_data['electricity_onsite_renew'], errors='coerce')
electricity_onsite_by_zipcode = ll84_data.groupby('Postal Code')['electricity_onsite_renew'].sum().reset_index()
electricity_onsite_by_zipcode.columns = ['zipcode', 'electricity_onsite_renew_50k']
electricity_onsite_by_zipcode['electricity_onsite_renew_per_50k_building'] = electricity_onsite_by_zipcode['electricity_onsite_renew_50k'] / ll84_buildings_df['building_count']
master_df = master_df.merge(electricity_onsite_by_zipcode[['zipcode', 'electricity_onsite_renew_per_50k_building']], on='zipcode', how='left')
# master_df['electricity_onsite_renew_50k'] = (master_df['electricity_onsite_renew_50k'].fillna(0))

# Green Power - Onsite (kWh)
ll84_data.rename(columns={'Green Power - Onsite (kWh)': 'green_power_onsite'}, inplace=True)
ll84_data['green_power_onsite'] = pd.to_numeric(ll84_data['green_power_onsite'], errors='coerce')
total_green_power_by_zipcode = ll84_data.groupby('Postal Code')['green_power_onsite'].sum().reset_index()
total_green_power_by_zipcode.columns = ['zipcode', 'green_power_onsite_50k']
total_green_power_by_zipcode['green_power_onsite_per_50k_building'] = total_green_power_by_zipcode['green_power_onsite_50k'] / ll84_buildings_df['building_count']
master_df = master_df.merge(total_green_power_by_zipcode[['zipcode', 'green_power_onsite_per_50k_building']], on='zipcode', how='left')
# master_df['green_power_onsite_50k'] = (master_df['green_power_onsite_50k'].fillna(0))

# Natural Gas – Weather Normalized Site Natural Gas Use (therms)
ll84_data.rename(columns={'Natural Gas - Weather Normalized Site Natural Gas Use (therms)': 'nat_gas_onsite'}, inplace=True)
ll84_data['nat_gas_onsite'] = pd.to_numeric(ll84_data['nat_gas_onsite'], errors='coerce')
total_nat_gas_by_zipcode = ll84_data.groupby('Postal Code')['nat_gas_onsite'].sum().reset_index()
total_nat_gas_by_zipcode.columns = ['zipcode', 'nat_gas_onsite_50k']
total_nat_gas_by_zipcode['nat_gas_onsite_per_50k_building'] = total_nat_gas_by_zipcode['nat_gas_onsite_50k'] / ll84_buildings_df['building_count']
master_df = master_df.merge(total_nat_gas_by_zipcode[['zipcode', 'nat_gas_onsite_per_50k_building']], on='zipcode', how='left')
# master_df['nat_gas_onsite_50k'] = (master_df['nat_gas_onsite_50k'].fillna(0))

#%% pluto

pluto_data = pd.read_csv('pluto_2020.csv')

pluto_data['postcode'] = (
    pluto_data['postcode']
    .dropna()                    # Remove NaNs
    .astype(float)              # Make sure it's float
    .astype(int)                # Drop the decimal
    .astype(str)                # Convert to string
    .str.zfill(5)               # Pad with leading zeros if needed
)

'''
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
'''

# number of buildings
num_buildings_by_zipcode = pluto_data.groupby('postcode')['numbldgs'].sum().reset_index()
num_buildings_by_zipcode.columns = ['zipcode', 'num_buildings']

# residential units per capita
total_res_units_by_zipcode = pluto_data.groupby('postcode')['unitsres'].sum().reset_index()
total_res_units_by_zipcode.columns = ['zipcode', 'total_res_units']
master_df = master_df.merge(total_res_units_by_zipcode, on='zipcode', how='left')
master_df['res_units_per_capita'] = master_df['total_res_units'] / master_df['population']
master_df = master_df.drop('total_res_units', axis=1)
# master_df['total_units'] = (master_df['total_units'].fillna(0))

# bldgarea
building_areas_by_zipcode = pluto_data.groupby('postcode')['bldgarea'].sum().reset_index()
building_areas_by_zipcode.columns = ['zipcode', 'total_building_areas']
building_areas_by_zipcode['average_building_area'] = building_areas_by_zipcode['total_building_areas'] / num_buildings_by_zipcode['num_buildings']
master_df = master_df.merge(building_areas_by_zipcode[['zipcode', 'average_building_area']], on='zipcode', how='left')
# master_df['total_building_areas'] = (master_df['total_building_areas'].fillna(0))

# borough
borough_mode = (
    pluto_data
      .groupby('postcode')['borough']
      .agg(lambda x: x.mode()[0])  # take the most frequent borough
      .reset_index()
)
borough_mode.columns = ['zipcode','borough']
master_df = master_df.merge(borough_mode, on='zipcode', how='left')


#%% community gardens

# get block to zipcode dataframe
block_zipcode_data = pluto_data[['postcode', 'Tax block']]
block_zipcode_data.columns = ['zipcode', 'block']

community_garden_data = pd.read_csv('greenthumb_community_gardens_2020.csv')
community_garden_data = community_garden_data.merge(block_zipcode_data, on='block', how='left')

# number of community gardens
community_gardens_by_zipcode = community_garden_data['zipcode'].value_counts().to_dict()
master_df['num_community_gardens'] = master_df['zipcode'].map(community_gardens_by_zipcode).fillna(0).astype(int)

#%% trees updated 

tree_data = pd.read_csv('street_tree_census_2015.csv')
tree_data = tree_data[['postcode', 'status', 'nta_name']]  # keeping nly needed columns
tree_data.rename(columns={'postcode':'zipcode'}, inplace=True)
tree_data['zipcode'] = tree_data['zipcode'].astype(str)

# filter to only ALIVE trees
alive_tree_data = tree_data[tree_data['status'] == 'Alive'].drop(columns=['status'])
alive_tree_data = alive_tree_data.dropna(subset=['zipcode'])

# count number of alive trees per zipcode
tree_counts_by_zipcode = alive_tree_data['zipcode'].value_counts().to_dict()

# map counts into master_df
master_df['num_trees'] = master_df['zipcode'].map(tree_counts_by_zipcode).fillna(0).astype(int)

# get neighborhoods associated with a zipcode
neighborhood_mode = (
    tree_data
      .groupby('zipcode')['nta_name']
      .agg(lambda x: x.mode()[0]) 
      .reset_index()
)
neighborhood_mode.columns = ['zipcode','neighborhood']
neighborhood_mode['zipcode'] = neighborhood_mode['zipcode'].astype(str)
master_df = master_df.merge(neighborhood_mode, on='zipcode', how='left')

#%% college dataset

# percent with 4 year degree
college_data = pd.read_csv('college_percentages_2021.csv')
college_data.columns = ['zipcode', 'college_percent']
college_data['college_percent'] = college_data['college_percent'] / 100
college_data['zipcode'] = college_data['zipcode'].astype(str)
master_df = master_df.merge(college_data, on='zipcode', how='left')

#%% justice dataset

# percent white, percent black, jail admissions per 100k
justice_data = pd.read_csv('justice_percentages_2021.csv')
justice_data.columns = ['zipcode', 'white_percent', 'black_percent', 'jail_admissions_per_100k']
justice_data['white_percent'] = pd.to_numeric(justice_data['white_percent'], errors='coerce')
justice_data['black_percent'] = pd.to_numeric(justice_data['black_percent'], errors='coerce')
justice_data['jail_admissions_per_100k'] = pd.to_numeric(justice_data['jail_admissions_per_100k'], errors='coerce')
justice_data['zipcode'] = justice_data['zipcode'].astype(str)
master_df = master_df.merge(justice_data[['zipcode', 'white_percent', 'black_percent', 'jail_admissions_per_100k']], on='zipcode', how='left')

#%% poverty dataset

# percent poverty
poverty_data = pd.read_csv('poverty_2020.csv')
poverty_data = poverty_data[['NAME','S1701_C01_002E', 'S1701_C01_001E']]
poverty_data = poverty_data[3:]
poverty_data = poverty_data.rename(columns={'NAME': 'zipcode'})
poverty_data = poverty_data.rename(columns={'S1701_C01_002E': 'poverty_estimate'})
poverty_data['zipcode'] = poverty_data['zipcode'].str.replace('ZCTA5 ', '', regex=False).astype(int)
poverty_data['poverty_percent'] = (
    poverty_data['poverty_estimate'].astype(int) / poverty_data['S1701_C01_001E'].astype(int)
).round(4)
poverty_data['zipcode'] = poverty_data['zipcode'].astype(str)
master_df = master_df.merge(poverty_data[['zipcode', 'poverty_percent']], on='zipcode', how='left')

#%% reorder columns of master_df for organization & drop rows with nan 

data = master_df[['zipcode', 'borough', 'neighborhood', 'population', 'average_building_area', 'res_units_per_capita',
                  'mean_energy_star_score_50k_building', 'electricity_onsite_renew_per_50k_building', 
                  'green_power_onsite_per_50k_building', 'nat_gas_onsite_per_50k_building', 'num_trees', 'total_acres_greenstreets', 'num_community_gardens', 
                  'white_percent', 'black_percent', 'poverty_percent', 'college_percent', 'jail_admissions_per_100k', 'income_group', 'hvi']]

data = data.dropna()

data.to_csv('hvi_final_data.csv', index=False)


