# -*- coding: utf-8 -*-

from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
os.getcwd()
# set wd
os.chdir('C:\\Users\\...')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the csv data using the Pandas library
data = 'arbejdsmarkedsanalyse_brancher.csv'
df = pd.read_csv(data, sep=';',encoding='latin-1', decimal=',')

# basic info
df.info()
df.describe()
df.corr()
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#%%
# looking for NaN's
# Extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns)
# how many missing values : 
df_count_nan = df[attributeNames].isnull().sum()

# Gennemsnit
df_gen_notnull = df[df['Gennemsnit'].notnull()]
# looks like the numbers for the different Groups overall 
df_Total = df_gen_notnull
# has 109 rows, should match 109 different Q 
len(np.unique(df[df['Question Label'].notnull()]['Question Label']))
# Values for Gennemsnit and Score (Total) are to be found for each 
# question.
len((df[df['Gennemsnit'].notnull()]['Gennemsnit']))
len((df[df['Score (Total)'].notnull()]['Score (Total)']))

same = ((df[df['Gennemsnit'].notnull()]['Gennemsnit']) ==
        (df[df['Score (Total)'].notnull()]['Score (Total)'])*
        np.sign(df[df['Score (Total)'].notnull()]['Hoej Score Godt']-0.5))
sum(same==False)
# Gennemsnit = Score (Total) pos/neg according to good score high or not (0/1)   

same = ((df[df['Group']== 'Total']['Score (Total) (Fixed)']) ==
        (df[df['Group']== 'Total']['Score (Total)']))
sum(same==False)
# Score(Total) Fixed = Score(Total) (for Group = Total)

same = ((df[df['Group']== 'Total']['Score (Indekseret score) (gennemsnit)']) ==
        (df[df['Group']== 'Total']['Score (Indekseret score)']))
sum(same==False)
# Score(indx score)(gennemsnit) = Gennemsnit (for Group = Total)


same = ((df[df['Group']== 'Total']['Score (Indekseret score) (gennemsnit)']) ==
        (df[df['Group']== 'Total']['Score (Indekseret score) (gennemsnit) (label)']))
sum(same==False)
# depended on Akse 

# look at field values index 
df_FVI_null = df[df['Field Values Index'].isnull()]
# looks like missing values for random Questions/Groups
# looking at the first row in df_FVI_null (Q_label = Arbejdsrelateret sygdom)
df_arb_syg = df[df['Question Label']=='Arbejdsrelateret sygdom'].sort_values(by=['Score'])
shape(df_arb_syg)
# 76 different Groups (incl. Total)
len(np.unique(df['Group']))

# Looks like a missing value
# All 'Field Values Index' isnull() removed from df 
df1 = df[df['Field Values Index'].notnull()]

# check antpct == score when Akse = andel (%)
df_andel = df1[df1['Akse']=='Andel (%)'][['Akse','Antpct','Score']]
sum((df_andel['Antpct']==df_andel['Score'])==False)

# check score = Field Values depending on Hoej score godt
same = (df1['Score'] == (np.sign(df1['Hoej Score Godt']-0.5))* df1['Field Values'])
sum(same==False)

# Field Values and Indekseret score depend on each other in different ways 

#%%
# Important for analyze
df2 = df1[attributeNames[[1,2,13,14,15,32,35,38]]]

# basic info
df2.info()
df2.describe()
df2.corr()
corrMatrix = df2.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Look into missing Q
uni, count = np.unique(df2['Group'], return_counts=True)
print(np.asarray((uni, count)).T)
print(len(uni[count == 109]))
print(len(np.unique(df2['Group'])))
# 24 out of 76 have answered all Q, incl. Total 

# roughly most and least represented 
df_antper = df2.sort_values(by=['Antpers'])
pd.unique(df_antper['Group'])[:4]
pd.unique(df_antper['Group'])[-5:]

# remove where Ordforklaring = null, do to no asked Q
df3 = df2[df2['Ordforklaring'].notnull()]
# check for avg. antpers 
df3[df3['Group']!='Total'].describe()


# Looking for outliers between Groups
df_group = df2[df2['Group']!='Total']

plt.figure('Box Antpers')
sns.boxplot(df_gender_age['Antpers'])
plt.xlabel('Number of pers for each question', fontsize = 14)
plt.title('Attribute Antpers', fontsize = 21)
plt.show()

#%%
### 'Arbejdsrelateret sygdom' ###
df_arb_syg = df3[df3['Question Label']=='Arbejdsrelateret sygdom'].sort_values(by=['Score'])
shape(df_arb_syg)

figure('Branche sygdom')
x_pos = range(len(df_arb_syg['Group']))
plt.xlabel('Group')
plt.ylabel('Andel(%)')
plt.title('Andel(%) by Group - Arbejdsrelateret sygdom')
plt.bar(x_pos,df_arb_syg['Score'])
# Rotation of the bars names
plt.xticks(x_pos,df_arb_syg['Group'], rotation=90)


#%%
### 'Løft, skub eller træk af byrder' ###
df_byrder = df3[df3['Topic Label']=='Løft, skub eller træk af byrder']

uni, count = np.unique(df_byrder_['Group'], return_counts=True)
print(np.asarray((uni, count)).T)
# few groups have answered all 4 Q

uni_byrder = np.unique(df_byrder['Question Label'])

# to be able to manage resulst only top/bottom 3 og heavy lift 
df_tung_sort = df_byrder[df_byrder['Question Label']== 
                         'Typiske løft på 30 kg eller derover'].sort_values(by=['Score'])
Groups_byrde = np.concatenate((np.asarray(df_tung_sort['Group'].head(3)),
                                         np.asarray(df_tung_sort['Group'].tail(3))))
# Groups with high/low score in 'Typiske løft på 30 kg eller derover'

df_byrder = df_byrder[(df_byrder['Group']== Groups_byrde[0]) |
                      (df_byrder['Group']== Groups_byrde[1]) |
                      (df_byrder['Group']== Groups_byrde[2]) |
                      (df_byrder['Group']== Groups_byrde[3]) |
                      (df_byrder['Group']== Groups_byrde[4]) |
                      (df_byrder['Group']== Groups_byrde[5]) ].sort_values(by=['Score'])
print(shape(df_byrder))

# just for 'Typiske løft på 30 kg eller derover'
figure('Typiske løft på 30 kg eller derover')
plt.bar(df_byrder[df_byrder['Question Label']== 
                         'Typiske løft på 30 kg eller derover']['Group'],
        df_byrder[df_byrder['Question Label']== 
                         'Typiske løft på 30 kg eller derover']['Score'])
plt.xlabel('Group')
plt.ylabel('Andel(%)')
plt.title('Andel(%) by Group - Typiske løft på 30 kg eller derover')

# antpers pr. Q
df_byrder[df_byrder['Question Label']== 
                         'Typiske løft på 30 kg eller derover'][['Group','Antpers']]


#%%
### 'Uoverskuelighed og stress' ###
df_stress = df2[df2['Topic Label']=='Uoverskuelighed og stress']


uni, count = np.unique(df_stress['Group'], return_counts=True)
print(np.asarray((uni, count)).T)
# Only ['Frisører og kosmetologer' 6] has not answered all Q
df_stress = df_stress[df_stress['Group']!='Frisører og kosmetologer']
len(np.unique(df_stress['Group']))

# only top/bottom 3 
df_stress_arb = df_stress[df_stress['Question Label']== 
                         'Arbejdsrelateret stress'].sort_values(by=['Score'])
Groups_strees_arb = np.concatenate((np.asarray(df_stress_arb['Group'].head(3)),
                                         np.asarray(df_stress_arb['Group'].tail(3))))

df_stress = df_stress[(df_stress['Group'] == Groups_strees_arb[0]) |
                      (df_stress['Group'] == Groups_strees_arb[1]) |
                      (df_stress['Group'] == Groups_strees_arb[2]) |
                      (df_stress['Group'] == Groups_strees_arb[3]) |
                      (df_stress['Group'] == Groups_strees_arb[4]) |
                      (df_stress['Group'] == Groups_strees_arb[5]) ].sort_values(by=['Score'])
print(shape(df_stress))

# For Arbejdsrelateret stress
figure('Arbejdsrelateret stress')
plt.bar(df_stress[df_stress['Question Label']== 
                         'Arbejdsrelateret stress']['Group'],
        df_stress[df_stress['Question Label']== 
                         'Arbejdsrelateret stress']['Score'])
plt.xlabel('Group')
plt.ylabel('Andel(%)')
plt.title('Andel(%) by Group - Arbejdsrelateret stress')
# antpers pr. Q
df_stress[df_stress['Question Label']== 
                         'Arbejdsrelateret stress'][['Group','Antpers']]
uni_Ord = np.unique(df_stress['Question Label'])

# data to plot
n_groups = len(np.unique(df_stress['Group']))

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.7

rects1 = plt.bar(index, df_stress[df_stress['Question Label']==uni_Ord[1]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='b',
                 label=uni_Ord[1])

rects2 = plt.bar(index + bar_width, df_stress[df_stress['Question Label']==uni_Ord[2]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='g',
                 label=uni_Ord[2])

rects3 = plt.bar(index + bar_width*2, df_stress[df_stress['Question Label']==uni_Ord[4]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='c',
                 label=uni_Ord[4])


rects4 = plt.bar(index + bar_width*3, df_stress[df_stress['Question Label']==uni_Ord[4]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='r',
                 label=uni_Ord[5])

rects5 = plt.bar(index + bar_width*4, df_stress[df_stress['Question Label']==uni_Ord[5]]['Score'],
                 bar_width,
                 alpha=opacity,
                 color='y',
                 label=uni_Ord[5])

plt.xlabel('Group')
plt.ylabel('Andel(%)')
plt.title('Andel(%) by Group')
plt.xticks(index + bar_width, pd.unique(df_stress[df_stress['Question Label']==uni_Ord[1]]['Group']))
plt.legend()
plt.tight_layout()
plt.show()