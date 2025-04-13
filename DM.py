# -*- coding: utf-8 -*-
"""
Created on Fri Apr 04 13:21:39 2025

@author: Karan
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

os.getcwd()


os.chdir("C:\\Users\\Karan\\Documents\\Project\\Data Mining")

Mood_Smartphone_df = pd.read_csv("dataset_mood_smartphone.csv")


#Removing this Unnamed column - Just had index number
Mood_Smartphone_df=Mood_Smartphone_df.drop(columns=["Unnamed: 0"])

# Sort by id and time
Mood_Smartphone_df=Mood_Smartphone_df.sort_values(by=['id', 'time'])

# Ignoring Time, just keeping date 
Mood_Smartphone_df['time'] = pd.to_datetime(Mood_Smartphone_df['time'])
Mood_Smartphone_df['time'] = Mood_Smartphone_df['time'].dt.date


pd.DataFrame(Mood_Smartphone_df.variable.value_counts()).plot.bar(title="Frequency of all features", legend=None,figsize=(10,7));








##################
##################
#Column wise data formation
##################
##################

# Defining variables by their aggregate function
mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence']
sum_vars = ['screen', 'call', 'sms', 'activity',
            'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
            'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
            'appCat.social', 'appCat.travel', 'appCat.unknown',
            'appCat.utilities', 'appCat.weather']

# Grouping of mood by average by date
Mood_meanByDate_df = Mood_Smartphone_df[Mood_Smartphone_df['variable'].isin(mean_vars)].groupby(
    ['id', 'time', 'variable']
)['value'].mean().reset_index()

# Grouping of other columns by sum by date
sum_df = Mood_Smartphone_df[Mood_Smartphone_df['variable'].isin(sum_vars)].groupby(
    ['id', 'time', 'variable']
)['value'].sum().reset_index()

# Combining both
combined_df = pd.concat([Mood_meanByDate_df, sum_df], ignore_index=True)

# Pivot to wide format
Mood_Smartphone_df2 = combined_df.pivot_table(
    index=['id', 'time'],
    columns='variable',
    values='value',
).reset_index()

#############
#Till here
#############   



##############
#Mood over time plots
##############

# How many individuals?
print(len(Mood_Smartphone_df2.id.unique()))

f = plt.figure(figsize=(30, 20))
num = 0

for i in Mood_Smartphone_df2['id'].unique():
    num += 1
    ax = f.add_subplot(5, 7, num)
    ax.set_title(f'ID: {i}')

    # Filter the data for the current ID
    df_id = Mood_Smartphone_df2[Mood_Smartphone_df2['id'] == i]

    # Plot mood over time
    ax.plot(df_id['time'], df_id['mood'], marker='', linewidth=2.8, alpha=0.9, label='mood')
    ax.plot(df_id['time'], df_id['mood'], marker='', color='grey', linewidth=0.9, alpha=0.3)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mood')

plt.tight_layout()
plt.show()


##############
##Till here
##############







#############
#############
#Correlation Matrix
#############
#############



df_corrrelation = Mood_Smartphone_df2.drop(['id', 'time'], axis=1)
correlation_matrix = df_corrrelation.corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



# New features (#Feature engineering)
Mood_Smartphone_df3=Mood_Smartphone_df2.copy()

Mood_Smartphone_df3['WorkApps'] = Mood_Smartphone_df2[['appCat.office', 'appCat.finance']].sum(axis=1, skipna=True)
Mood_Smartphone_df3['FunApps'] = Mood_Smartphone_df2[['appCat.game','appCat.entertainment','appCat.social']].sum(axis=1, skipna=True)
Mood_Smartphone_df3['ConvenientApps'] = Mood_Smartphone_df2[['appCat.travel','appCat.utilities','appCat.weather']].sum(axis=1, skipna=True)
Mood_Smartphone_df3['UnknownApps'] = Mood_Smartphone_df2[['appCat.other','appCat.unknown']].sum(axis=1, skipna=True)
column_names = ['appCat.builtin','appCat.entertainment','appCat.finance','appCat.game','appCat.office','appCat.other','appCat.social','appCat.travel','appCat.unknown','appCat.utilities','appCat.weather']
Mood_Smartphone_df3 = Mood_Smartphone_df3.drop(columns=column_names, axis=1)



df_corrrelation = Mood_Smartphone_df3.drop(['id', 'time'], axis=1)
correlation_matrix = df_corrrelation.corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


##################
#Till here########
##################




##################
#Distribution of features
##################


def univariate_cont_analyis(*columns):
    plt.figure(figsize = (20,20))
    
    for i, col in enumerate(columns):
        plt.subplot(5,3,i+1)
        sns.histplot(Mood_Smartphone_df2[col], color = 'r', kde = True, label = col)
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.tight_layout()



univariate_cont_analyis("mood","screen","activity","call","sms","circumplex.arousal",
                        "circumplex.valence","appCat.entertainment","appCat.social")



# For engineered features
def univariate_cont_analyis(*columns):
    plt.figure(figsize = (20,20))
    
    for i, col in enumerate(columns):
        plt.subplot(5,3,i+1)
        sns.histplot(Mood_Smartphone_df3[col], color = 'r', kde = True, label = col)
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.tight_layout()

univariate_cont_analyis("mood","WorkApps","FunApps","ConvenientApps","UnknownApps","appCat.communication")




###############
#Missing values, still woriking on it
###############

Mood_Smartphone_df2.isna().sum()


first_mood_date = Mood_Smartphone_df3.loc[Mood_Smartphone_df2['mood'].first_valid_index(), 'time']
print("First date with mood value:", first_mood_date)




starting_dates = {}

for col in Mood_Smartphone_df2.columns:
    first_valid_idx = Mood_Smartphone_df2[col].first_valid_index()
    if first_valid_idx is not None and 'time' in Mood_Smartphone_df2.columns:
        starting_dates[col] = Mood_Smartphone_df2.loc[first_valid_idx, 'time']
    else:
        starting_dates[col] = None

starting_dates_df = pd.DataFrame.from_dict(starting_dates, orient='index', columns=['First Non-NaN Date'])
starting_dates_df.index.name = 'Column'
print(starting_dates_df)




# Ensure time column is datetime and sorted
Mood_Smartphone_df2['time'] = pd.to_datetime(Mood_Smartphone_df2['time'])
Mood_Smartphone_df2 = Mood_Smartphone_df2.sort_values(by=['id', 'time'])

# Create a copy to modify
clean_df = Mood_Smartphone_df2.copy()

# Loop over each user
for uid, group in Mood_Smartphone_df2.groupby('id'):
    group = group.set_index('time').sort_index()
    mood_series = group['mood']

    streak_count = 0
    streak_start = None

    for date, value in mood_series.items():
        if pd.isna(value):
            if streak_count == 0:
                streak_start = date
            streak_count += 1
        else:
            if streak_count > 3:
                # Remove the streak from clean_df
                dates_to_drop = pd.date_range(start=streak_start, end=prev_date)
                clean_df = clean_df.drop(index=clean_df[(clean_df['id'] == uid) & (clean_df['time'].isin(dates_to_drop))].index)
            streak_count = 0
            streak_start = None
        prev_date = date

    # Check if last values in group are NaNs
    if streak_count > 3:
        dates_to_drop = pd.date_range(start=streak_start, end=prev_date)
        clean_df = clean_df.drop(index=clean_df[(clean_df['id'] == uid) & (clean_df['time'].isin(dates_to_drop))].index)

# Optional: reset index if needed
clean_df = clean_df.reset_index(drop=True)


missing_info = pd.DataFrame({
    'Missing Count': clean_df.isna().sum(),
    'Missing %': clean_df.isna().mean() * 100
})
print(missing_info)



first_mood_dates = (
    clean_df
    .loc[clean_df['mood'].notna()]
    .sort_values(by=['id', 'time'])
    .groupby('id')
    .first()
    .reset_index()[['id', 'time']]
)

print(first_mood_dates)







