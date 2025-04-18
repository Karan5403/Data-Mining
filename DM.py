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
mean_vars = ['mood', 'circumplex.arousal', 'activity','circumplex.valence']
sum_vars = ['screen', 'call', 'sms', 
            'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
            'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
            'appCat.social', 'appCat.travel', 'appCat.unknown',
            'appCat.utilities', 'appCat.weather']



# Grouping of mood by average by date
Mean_df = Mood_Smartphone_df[Mood_Smartphone_df['variable'].isin(mean_vars)].groupby(
    ['id', 'time', 'variable'])['value'].mean().reset_index()

# Grouping of other columns by sum by date
sum_df = Mood_Smartphone_df[Mood_Smartphone_df['variable'].isin(sum_vars)].groupby(
    ['id', 'time', 'variable'])['value'].sum().reset_index()

# Combining both
combined_df = pd.concat([Mean_df, sum_df], ignore_index=True)

# Pivot to wide format
Mood_Smartphone_df2 = combined_df.pivot_table(
    index=['id', 'time'],
    columns='variable',
    values='value',
).reset_index()


abc=Mood_Smartphone_df2.describe()

#these is/are negative values in appCat.builtin, which is not possible

counter=0
for i in range(len(Mood_Smartphone_df2)):
    if Mood_Smartphone_df2['appCat.builtin'][i]<0:
        counter+=1
        Mood_Smartphone_df2['appCat.builtin'][i]=np.nan
print('Values changed: {}'.format(counter))


#############
#Till here
#############   


###########
#Outliers
##########
fig = plt.figure(figsize = (15,5))

for i in Mood_Smartphone_df2.columns:
    sns.boxplot(x = Mood_Smartphone_df2[i], data = Mood_Smartphone_df2, palette = 'rainbow', orient = 'h')
    print('Feature : ',i)
    plt.show()

# Treatment not done directly
# Later in program scaling helps to treat them 

#############
#Till here
#############   




###############
## Adding t column for each id  to track the sequence of days in a structured and numeric way.
###############
Mood_Smartphone_df['time'] = pd.to_datetime(Mood_Smartphone_df['time'])
dates = pd.date_range(
    start=Mood_Smartphone_df['time'].min().normalize(),
    end=Mood_Smartphone_df['time'].max().normalize(),
    freq='D'
)

ids = Mood_Smartphone_df2.id.unique()


df_dates = pd.DataFrame({'time': dates})

for i in range(len(df_dates)):
    df_dates.at[i, 't'] = i + 1


Mood_Smartphone_df2['time'] = pd.to_datetime(Mood_Smartphone_df2['time'])
df_dates['time'] = pd.to_datetime(df_dates['time'])
Mood_Smartphone_df2 = Mood_Smartphone_df2.merge(df_dates[['time', 't']], on='time', how='left')



#############
#Till here
#############   



##############
#Mood over time plots
# plotted 2 ids separately for report

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
    ax.plot(df_id['time'], df_id['mood'], marker='', linewidth=2, alpha=0.5, label='mood')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mood')

plt.tight_layout()
plt.show()




# Filter only the desired IDs
selected_ids = ['AS14.14', 'AS14.29']
filtered_df = Mood_Smartphone_df2[Mood_Smartphone_df2['id'].isin(selected_ids)]

# Plotting
f = plt.figure(figsize=(15, 8))  # Smaller figure since only 2 plots
for num, i in enumerate(selected_ids, 1):
    ax = f.add_subplot(1, 2, num)
    ax.set_title(f'ID: {i}')
    df_id = filtered_df[filtered_df['id'] == i]
    ax.plot(df_id['time'], df_id['mood'], marker='', linewidth=2, alpha=0.5, label='mood')
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



# New features (#Feature engineering 1)
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
#Distribution of old and new features
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
#Missing values
###############

Mood_Smartphone_df3.isna().sum()


#Cleaning up long streaks of missing values (NaNs) in the mood column
#if a user has more than 3 consecutive days with missing mood values,
# you drop that entire streak for that user from the dataset.


# Ensure time column is datetime and sorted
Mood_Smartphone_df3['time'] = pd.to_datetime(Mood_Smartphone_df3['time'])
Mood_Smartphone_df3 = Mood_Smartphone_df3.sort_values(by=['id', 'time'])

# Create a copy to modify
clean_df = Mood_Smartphone_df3.copy()

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


clean_df = clean_df.reset_index(drop=True)


dfdescribe= clean_df.describe()



Mood_Smartphone_df3.isna().sum()
clean_df.isna().sum()


missing_info = pd.DataFrame({
    'Missing Count': clean_df.isna().sum(),
    'Missing %': clean_df.isna().mean() * 100
})
print(missing_info)



### For call sms, its either yes or no,, for no, there are NAs,  so changed them to 0
clean_df[['call', 'sms']] = clean_df.groupby('id')[['call', 'sms']].transform(lambda x: x.fillna(0))
clean_df.isna().sum()

#  fill NaN values in the "activity" column using the mean for each id and for a specific month (from the time column)
# if group meaan not available, then just replace by 0


clean_df['month'] = clean_df['time'].dt.to_period('M')
clean_df['activity'] = clean_df.groupby(['id', 'month'])['activity']\
    .transform(lambda x: x.fillna(x.mean() if not x.mean() != x.mean() else 0))

### For screen time, and appCat.communications,,, missing values taken as 0.
clean_df[['appCat.communication', 'screen']] = clean_df.groupby('id')[['appCat.communication', 'screen']].transform(lambda x: x.fillna(0))
clean_df.isna().sum()




####
# For Missing values in mood, Arousal, valence we use interpolate function
####

interpolated_df = clean_df.copy()

# Interpolate only the desired numeric columns
interpolated_df[['mood', 'circumplex.arousal', 'circumplex.valence']] = (
    interpolated_df[['mood', 'circumplex.arousal', 'circumplex.valence']].interpolate(axis=0)
)

interpolated_df=interpolated_df.drop(columns=["month"])

interpolated_df.isna().sum()






f = plt.figure(figsize=(30, 20))
num = 0

for i in Mood_Smartphone_df2['id'].unique():
    num += 1
    ax = f.add_subplot(5, 7, num)
    ax.set_title(f'ID: {i}')
    # Filter the data for the current ID
    df_id = clean_df[clean_df['id'] == i]
    df_id_interp = interpolated_df[interpolated_df['id'] == i]
    # Plot mood over time
    ax.plot(df_id['time'], df_id['mood'], marker='', linewidth=2, alpha=0.8, label='mood',color ='blue')
    ax.plot(df_id_interp['time'], df_id_interp['mood'], linewidth=2, alpha=0.5, label='Interpolated Mood', color='red', linestyle=':')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mood')

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt

# Specify the selected IDs
selected_ids = ['AS14.14', 'AS14.29']

# Create figure
f = plt.figure(figsize=(15, 8))  # Adjusted size for 2 subplots
for num, i in enumerate(selected_ids, 1):
    ax = f.add_subplot(1, 2, num)
    ax.set_title(f'ID: {i}')
    
    # Filter data
    df_id = clean_df[clean_df['id'] == i]
    df_id_interp = interpolated_df[interpolated_df['id'] == i]
    
    # Plot original mood
    ax.plot(df_id['time'], df_id['mood'], marker='', linewidth=2, alpha=0.8, label='Mood', color='blue')
    
    # Plot interpolated mood
    ax.plot(df_id_interp['time'], df_id_interp['mood'], linewidth=2, alpha=0.5, 
            label='Interpolated Mood', color='red', linestyle=':')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mood')
    ax.legend()

plt.tight_layout()
plt.show()


rows_to_drop = [
    ('AS14.12', '2014-03-15'),
    ('AS14.01', '2014-02-26'),
    ('AS14.01', '2014-02-27')
]

interpolated_df['time'] = interpolated_df['time'].astype(str)
interpolated_df = interpolated_df[
    ~interpolated_df[['id', 'time']].apply(tuple, axis=1).isin(rows_to_drop)
]

####
####
#Will use temporal later for RNN
####
temporal_df=interpolated_df.copy()

####################
# FEATURE ENGENEERING Creating 5 day window for all variables
####################

nontemporal_df = interpolated_df.copy()

nontemporal_df = nontemporal_df.reset_index().set_index(['id', 't'])
nontemporal_df=nontemporal_df.drop(columns=["index"])
# List of variables to apply rolling average to
new_variables = [col for col in nontemporal_df.columns if col != 'time' and col != 'id']


# Compute 5-day rolling average for each variable
for var in new_variables:
    # New column name
    new_col = f'{var}_Agg_5day'
    # Rolling average grouped by ID
    nontemporal_df[new_col] = nontemporal_df.groupby('id')[var].transform(lambda x: x.shift(1).rolling(5).mean())

columns_to_drop = [col for col in new_variables if col != 'mood']

# Drop the columns from the dataframe
nontemporal_df = nontemporal_df.drop(columns=columns_to_drop)


###########
##Till here
###########


#############
##Scaling
#############


from sklearn.preprocessing import RobustScaler, MinMaxScaler

scaled = nontemporal_df.copy()
scaled.reset_index(inplace=True)
scaled.set_index(['id','time','t'],inplace=True,drop=True)

columns_to_scale = [col for col in scaled.columns]


minmaxscaler = MinMaxScaler()
scaled[columns_to_scale] = minmaxscaler.fit_transform(scaled[columns_to_scale])
    
scaled=scaled.drop(columns=["mood_Agg_5day"])

scaled.to_csv('nontemporal.csv')



scaled = scaled.groupby(level=0).apply(lambda group: group.iloc[5:]).reset_index(level=0, drop=True)


###########
#Till here
###########



############
##Modelling 
############



############
##RF Regressor 
############

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


X = scaled.iloc[:,np.arange(1,len(scaled.columns)).tolist()]
y = scaled['mood']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2)


rf = RandomForestRegressor(random_state=2)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [10, 20,40,80],
    'min_samples_split': [2,4],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # use all processors
    verbose=2
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best model and parameters
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate on test set
y_pred = best_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.3f}")


########
#Best Parameters
########

#Best parameters: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
#Test RMSE: 0.754

model = RandomForestRegressor(max_depth=40, max_features = 'sqrt',min_samples_leaf=1, n_estimators = 1000,
                               min_samples_split=4, random_state=2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)





from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predictions already made
# y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# Print
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")







y_pred_series_full = pd.Series(y_pred, index=y_test.index, name='y_pred')


# Get first ID from the MultiIndex
first_id = "AS14.14"

# unscaled
y_test1 = np.repeat(y_test.values.reshape(-1,1), repeats=scaled.shape[1]+1,axis=1)
y_test_unscaled = minmaxscaler.inverse_transform(y_test1)[:,0]


y_pred1 = np.repeat(y_pred.reshape(-1,1), repeats=scaled.shape[1]+1,axis=1)
y_pred_unscaled = minmaxscaler.inverse_transform(y_pred1)[:,0]


# Metrics for unscaled data
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
rmse = np.sqrt(mse)


# Print
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")





# Filter y_test and y_pred for only that ID
id_mask = y_test.index.get_level_values('id') == first_id
time_index = y_test.index.get_level_values('time')[id_mask]

# Create Series with time index for that ID
y_test_series = pd.Series(y_test_unscaled[id_mask], index=time_index)
y_pred_series = pd.Series(y_pred_unscaled[id_mask], index=time_index)



# Sort both by time
y_test_sorted = y_test_series.sort_index()
y_pred_sorted = y_pred_series.sort_index()

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_sorted.index, y_test_sorted, label='Actual', linewidth=2)
plt.plot(y_pred_sorted.index, y_pred_sorted, label='Predicted', linestyle='--', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Target Value')
plt.title(f'Actual vs Predicted for ID: {first_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


############
##RF Regressor till here
############





############
## RF for classification
############



scaled_classification = nontemporal_df.copy()

def classify_mood(mood_score):
    if mood_score <= 7:
        return 'Low'
    else:
        return 'High'

# Apply the function to the 'mood' column

scaled_classification['mood_classification'] = scaled_classification['mood'].apply(classify_mood)




def univariate_cont_analyis(*columns):
    plt.figure(figsize = (20,20))
    
    for i, col in enumerate(columns):
        plt.subplot(5,3,i+1)
        sns.histplot(scaled_classification[col], color = 'r', kde = True, label = col)
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.tight_layout()



univariate_cont_analyis('mood','mood_classification')

scaled_classification=scaled_classification.drop(columns=["mood"])



from sklearn.preprocessing import RobustScaler, MinMaxScaler

scaled_classification.reset_index(inplace=True)
scaled_classification.set_index(['id','time','t'],inplace=True,drop=True)

columns_to_scale = [col for col in scaled_classification.columns if col != 'mood_classification']


minmaxscaler = MinMaxScaler()
scaled_classification[columns_to_scale] = minmaxscaler.fit_transform(scaled_classification[columns_to_scale])
    
scaled_classification=scaled_classification.drop(columns=["mood_Agg_5day"])

scaled_classification.to_csv('nontemporal_classification.csv')

scaled_classification = scaled_classification.groupby(level=0).apply(lambda group: group.iloc[5:]).reset_index(level=0, drop=True)





#############
##MODEL
#############


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


X = scaled_classification.iloc[:, np.arange(1, len(scaled_classification.columns) - 1)]  # exclude 'mood' and 'mood_class'
y = scaled_classification['mood_classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)



clf = RandomForestClassifier(random_state=2)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10,20,40,60],
    'min_samples_split': [1,2,4],
    'min_samples_leaf': [1,2,4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))





########
#Best Parameters
########

#Best parameters: {'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}


model = RandomForestClassifier(max_depth=10, max_features = 'sqrt',min_samples_leaf=2, n_estimators = 300,
                               min_samples_split=2, random_state=2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Precision, Recall, F1 for multiclass (macro, micro, weighted available)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print them
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: full report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



##########
##Till here
##########




############
## RNN
############


temporal_df=interpolated_df.copy()

from sklearn.preprocessing import MinMaxScaler

scaledRNN = temporal_df.copy()
scaledRNN.reset_index(inplace=True)
scaledRNN.set_index(['id','time','t'],inplace=True,drop=True)

columns_to_scale = [col for col in scaledRNN.columns]


minmaxscaler = MinMaxScaler()

scaledRNN[columns_to_scale] = minmaxscaler.fit_transform(scaledRNN[columns_to_scale])


scaledRNN.reset_index(inplace=True)
scaledRNN=scaledRNN.drop(columns=["index",'t'])    
scaledRNN.to_csv('temporal.csv')




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense ,Dropout
from scikeras.wrappers import KerasRegressor
from scikeras.wrappers import KerasClassifier


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import train_test_split


scaledRNN['time'] = pd.to_datetime(scaledRNN['time'])
scaledRNN = scaledRNN.sort_values(by=['id', 'time'])

# --- Feature Engineering ---
features = scaledRNN.drop(columns=['id', 'time', 'mood'])
target = scaledRNN['mood']
ids = scaledRNN['id']
times = scaledRNN['time']


# --- Create sequences across all users ---
SEQ_LEN = 3

def create_sequences(X, y, time, ids, seq_len=SEQ_LEN):
    Xs, ys, ts, id_seq = [], [], [], []
    for i in range(len(X) - seq_len):
        if ids[i] == ids[i + seq_len]:  # ensure same user in sequence
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
            ts.append(time[i+seq_len])
            id_seq.append(ids[i+seq_len])
    return np.array(Xs), np.array(ys), np.array(ts), np.array(id_seq)

X, y, ts, ids_seq = create_sequences(features, target.values, times.values, ids.values)

# --- Global random train-test split (users mixed in both) ---
X_train, X_test, y_train, y_test, ts_train, ts_test, ids_train, ids_test = train_test_split(
    X, y, ts, ids_seq, test_size=0.2, shuffle=True, random_state=42
)


# --- Build and Train Model ---
model = Sequential([
    LSTM(128, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)



# --- Predict on test set ---
preds = model.predict(X_test)



# unscaled
y_test1 = np.repeat(y_test.reshape(-1,1), repeats=scaled.shape[1]+1,axis=1)
y_test_unscaled = minmaxscaler.inverse_transform(y_test1)[:,6]


y_pred1 = np.repeat(preds.reshape(-1,1), repeats=scaled.shape[1]+1,axis=1)
y_pred_unscaled = minmaxscaler.inverse_transform(y_pred1)[:,6]


# --- Plot predictions separately per user ---
unique_ids = np.unique(ids_test)

for uid in unique_ids:
    idx = ids_test == uid
    if np.sum(idx) < 5:
        continue  # Skip small groups

    # Sort data by time
    sorted_indices = np.argsort(ts_test[idx])
    ts_sorted = ts_test[idx][sorted_indices]
    y_sorted = y_test_unscaled[idx][sorted_indices]
    preds_sorted = y_pred_unscaled[idx][sorted_indices]

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(ts_sorted, y_sorted, label='Actual Mood', marker='o')
    plt.plot(ts_sorted, preds_sorted, label='Predicted Mood', marker='x')
    plt.title(f'Mood Prediction for User {uid} (Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Mood')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    

# Evaluate predictions
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}") 
    
    
    
    
   
    
    
    
