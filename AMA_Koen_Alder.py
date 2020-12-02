# -*- coding: utf-8 -*-

from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
os.getcwd()
# set wd to file
os.chdir('C:\\Users\\...')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the csv data using the Pandas library
data = 'arbejdsmarkedsanalyse_koen_alder.csv'
df = pd.read_csv(data, sep=';',encoding='latin-1', decimal=',')

### basic info ###
df.info()
# 'Gennemsnit', 'Score_Indx_score_mean_label' and 'Score_Indx_score_mean' 
# are w/out values

df.describe()
df.corr()
# visualization of corr
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns)

# how many missing values : 
df_count_nan = df[attributeNames].isnull().sum()

# Look into 'Ordforklaring' 
df_Ord_null = df[df['Ordforklaring'].isnull()]
np.unique(df_Ord_null['Question_Label'])
np.unique(df_Ord_null['Akse'])

# Ordf. vs. Q_label
len(np.unique(df1['Question_Label'])) == len(np.unique(df1['Ordforklaring']))
num_Q = len(np.unique(df1['Question_Label']))

#%%
# Removed from total df to count avg. antpers pr Q
df1 = df[df['Ordforklaring'].notnull()]

# representation
uni, count = np.unique(df1['Group'], return_counts=True)
print(np.asarray((uni, count)).T)

# count avg. antpers pr Q
avg_num = round(df1.groupby('Group')['Antpers'].sum()/num_Q,1)
type(avg_num) # because it is a Series ->
df_avg_num = avg_num.reset_index() # convert to df
df_avg_num = df_avg_num.rename(columns = {'Antpers':'Avg. pers'}) # change attribute name

# Visualization of distribution of gender and age group on an avf. question
plt.figure('Avg. pers Køn/Alder')
plt.bar(df_avg_num['Group'],df_avg_num['Avg. pers'])
plt.xlabel('Group')
plt.ylabel('Avg. pers')
plt.title('Avg. pers pr Question by Group')
plt.show()
## on average more women than men 

# Looking for outliers between gender/age groups
df_gender_age = df1[(df1['Group']!='Kvinder') & (df1['Group']!='Mænd')]

plt.figure('Box Antpers')
sns.boxplot(df_gender_age['Antpers'])
plt.xlabel('Number of pers for each question', fontsize = 14)
plt.title('Attribute Antpers', fontsize = 21)
plt.show()
## no group is under/over reprented 

#%%
# Look into Topic_Label 
np.unique(df['Topic_Label'])
# Arb. relateret sygdom 
df_arb_syg = df[df['Question_Label']== 'Arbejdsrelateret sygdom']
plt.figure('Arbejdsrelateret sygdom - Køn/Alder')
plt.bar(df_arb_syg['Group'],df_arb_syg['Score'])
plt.xlabel('Group')
plt.ylabel('Andel (%)')
plt.title('Avg. pers pr Question by Group')
plt.show()

#%%
# Looking into stress and missing 'Question_Label'
df_stress = df[df['Topic_Label'] == 'Uoverskuelighed og stress']

# percentage distribution of stress for stressed people 
uni_Ord_null = np.unique(df_Ord_null['Question_Label'])

# data to plot
n_groups = len(np.unique(df['Group']))

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.7

rects1 = plt.bar(index, df_Ord_null[df_Ord_null['Question_Label']==uni_Ord_null[0]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='b',
                 label=uni_Ord_null[0])

rects2 = plt.bar(index + bar_width, df_Ord_null[df_Ord_null['Question_Label']==uni_Ord_null[1]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='g',
                 label=uni_Ord_null[1])

rects3 = plt.bar(index + bar_width*2, df_Ord_null[df_Ord_null['Question_Label']==uni_Ord_null[2]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='y',
                 label=uni_Ord_null[2])

rects4 = plt.bar(index + bar_width*3, df_Ord_null[df_Ord_null['Question_Label']==uni_Ord_null[3]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='r',
                 label=uni_Ord_null[3])

rects5 = plt.bar(index + bar_width*4, df_Ord_null[df_Ord_null['Question_Label']==uni_Ord_null[4]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='m',
                 label=uni_Ord_null[4])

plt.xlabel('Group')
plt.ylabel('Andel(%)')
plt.title('Andel(%) by Group')
plt.xticks(index + bar_width, pd.unique(df_Ord_null[df_Ord_null['Question_Label']==uni_Ord_null[0]]['Group']))
plt.legend()
plt.tight_layout()
plt.show()
#%%
# Look into how many have answered Q about stress 
# Stress questions
np.unique(df_stress['Question_Label'])
# w/out 'personer med stress' (w/ ordforklaring)
q_stress = np.unique(df_stress['Question_Label'])[[0,3,7]]
# Look at ordforklaring 
print(np.unique(df_stress[(df_stress['Question_Label']==q_stress[0]) 
                    | (df_stress['Question_Label']==q_stress[1])
                    | (df_stress['Question_Label']==q_stress[2])]['Ordforklaring']))

df_stress_Ord = df_stress[(df_stress['Question_Label']==q_stress[0]) 
                          | (df_stress['Question_Label']==q_stress[1])
                          | (df_stress['Question_Label']==q_stress[2])]
# group repres.
avg_num_stress = round(df_stress_Ord.groupby('Group')['Antpers'].sum()/3,1)
print(avg_num_stress)

# Look at score for stress and 'overskuelighed' 
df_stress_over = df_stress[(df_stress['Question_Label']==q_stress[1]) 
                           | (df_stress['Question_Label']==q_stress[2])]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.7

rects1 = plt.bar(index, df_stress_over[df_stress_over['Question_Label']==q_stress[1]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='b',
                 label=q_stress[1])
rects2 = plt.bar(index + bar_width, df_stress_over[df_stress_over['Question_Label']==q_stress[2]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='r',
                 label=q_stress[2])

plt.xlabel('Group')
plt.ylabel('Score')
plt.title('Score by Group - Stress')
plt.xticks(index + bar_width, pd.unique(df_stress_over[df_stress_over['Question_Label']==q_stress[1]]['Group']))
plt.legend()
plt.tight_layout()
plt.show()


# 'Arbejdsrelateret stress'
df_stress_arb = df_stress[df_stress['Question_Label']=='Arbejdsrelateret stress']
plt.figure('Arb. Stress')
plt.bar(df_stress_arb['Group'],df_stress_arb['Score'])
plt.title('Arbejdsrelateret stress')
plt.xlabel('Group')
plt.ylabel('Andel (%)')
plt.show()

#%%
# Looking into 'Løft, skub eller træk af byrder'
df_byrder = df[df['Topic_Label']=='Løft, skub eller træk af byrder']
uni_byrder = np.unique(df_byrder['Question_Label'])

# data to plot
n_groups = len(np.unique(df['Group']))

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.7

rects1 = plt.bar(index, df_byrder[df_byrder['Question_Label']==uni_byrder[0]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='b',
                 label=uni_byrder[0])

rects2 = plt.bar(index + bar_width,df_byrder[df_byrder['Question_Label']==uni_byrder[1]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='g',
                 label=uni_byrder[1])

rects3 = plt.bar(index + bar_width*2,df_byrder[df_byrder['Question_Label']==uni_byrder[2]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='y',
                 label=uni_byrder[2])

rects4 = plt.bar(index + bar_width*3,df_byrder[df_byrder['Question_Label']==uni_byrder[3]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='r',
                 label=uni_byrder[3])

plt.xlabel('Group')
plt.ylabel('Andel(%)')
plt.title('Andel(%) by Group - Løft, skub eller træk af byrder')
plt.xticks(index + bar_width, pd.unique(df_byrder[df_byrder['Question_Label']==uni_byrder[0]]['Group']))
plt.legend()
plt.tight_layout()
plt.show()







