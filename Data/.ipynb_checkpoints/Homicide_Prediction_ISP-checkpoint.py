# standard external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn functions
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# regression model
from sklearn.ensemble import RandomForestRegressor


# load dataset
file_location ='database_var.csv'
df = pd.read_csv(file_location)


# obtain dataset information and some ststistical measures
print(df.describe())

df.info()

# keep needed columns
data = df[['Year', 'Geography', 'Data.Type', 'Homicide','Attempted.Murder', 'Sexual.Assault..Level.1', 'Assault..Level.1', 'Criminal.Harassment', 'Uttering.Threats']]

#total count of NaNs
for idx,col in enumerate(data.isnull().sum()):
    print(data.columns[idx],col)
    
provinces = ['Prince Edward Island','Newfoundland and Labrador', 'Nova Scotia', 'New Brunswick', 'Quebec', 'Ontario', 'Manitoba',
            'Saskatchewan', 'Alberta', 'British Columbia', 'Yukon', 'Northwest Territories', 'Yukon']  

subset_provinces = data[data['Geography'].isin(provinces)  & (data['Data.Type'].isin(['actual_incidents']))]
subset_provinces


# plot homicide trend in bar chart
plt.figure(figsize= (6,4), dpi = 150)
subset_provinces.groupby('Year')['Homicide'].sum().plot(kind='bar', color= 'lightgreen')
# Add labels and title
plt.xlabel('Year')
plt.ylabel('Number of Homicide')
plt.title('Homicide Trend in Canada')

#set the features and target
features = ['Attempted.Murder', 'Sexual.Assault..Level.1', 'Assault..Level.1', 'Criminal.Harassment', 'Uttering.Threats', 'Geography']
X1 = subset_provinces[features]
y = subset_provinces.Homicide

# show violent crime number distribution in the data set 
# Define the number of rows and columns you want
n_rows=2
n_cols=3
# inclue only violent crimes in the variables for distribution display
features_crime = features[:-1]
# Create the subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
fig.set_size_inches(15, 5)
for i, column in enumerate(features_crime):
    sns.histplot(subset_provinces[column], ax=axes[i//n_cols, i % n_cols], kde=True)
plt.tight_layout()
# Removing empty subplot
fig.delaxes(axes[1,2])


# create bar chart subplots for homicide rates for each province/territory
plt.figure(figsize= (15,8), dpi = 150)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=1)
for i , (Geography, data) in enumerate(subset_provinces.groupby("Geography")):
    ax = plt.subplot(3, 4, i+1)
    pivot = (data.groupby("Year")["Homicide"].sum().plot(kind='bar', color= 'lightgreen'))
    ax.set_title(Geography)  



# Create heatmap to visualize homicide rates in Canada
# create correlation matrix
corr_matrix = subset_provinces.pivot_table(index="Geography", columns="Year", values="Homicide")
# create heatmap
plt.figure(figsize= (6,4), dpi = 150)
sns.heatmap(corr_matrix, cmap="coolwarm")
# Add title
plt.title("Canada Homicide Heatmap") 
# Show plot
plt.show()


# encode category feature using multilabelbinarizer
mlb = MultiLabelBinarizer()
encoded = mlb.fit_transform(X1['Geography'])
encoded_X = pd.DataFrame(encoded, columns=mlb.classes_, index=X1['Geography'].index)
# Drop old column and merge new encoded columns
X2 = X1.drop('Geography', axis=1)
X = pd.concat([X2, encoded_X], axis=1, sort=False)


# split for training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# define random forest regressor model
from sklearn.ensemble import RandomForestRegressor
number_of_trees = 100 # set the number of trees in the forest
model = RandomForestRegressor(n_estimators = number_of_trees, random_state = 42)
# fit model
model.fit(X_train,y_train)


# predict test data
y_pred = model.predict(X_test)
#calculate the mean squared error
pred_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# calculate R^2                 
pred_r2 = r2_score(y_test, y_pred)
# print performance metrics                 
print("Root mean square is:", pred_rmse)
print("\nR2 score is:", pred_r2)


# calculate error percent
print(pred_rmse/np.mean(y_train))

# show how the model fits the test data
plt.figure(dpi=150)
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color = 'blue')
plt.title("Actual Homicide vs Predicted Homicide")
plt.xlabel('Actual Homicide')
plt.ylabel('Predicted Homicide')
# display plot
plt.show()