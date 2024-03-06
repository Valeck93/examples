# Import relevant packages / libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
%matplotlib inline
sns.set_theme(style="whitegrid")

# Load in both World Happiness Report datasets
df1 = pd.read_csv("world-happiness-report.csv")
df2 = pd.read_csv("world-happiness-report-2021.csv")

# Display number of columns and rows per dataset
df_colrows = pd.DataFrame(data={"Number of columns":[len(df1.columns), len(df2.columns)], "Number of rows":[len(df1), len(df2)]}, index=["WHI", "WHI2021"])
display(df_colrows)
print("World Happiness Report:", len(df1.columns), "columns and", len(df1), "rows.")
print("World Happiness Report 2021:", len(df2.columns), "columns and", len(df2), "rows.")

# List column names of both datasets
df1_col_list = df1.columns.tolist()
df2_col_list = df2.columns.tolist()

display(df1_col_list)
print("\n")
display(df2_col_list)

# Check for rates of missing values
display(df1.isna().sum()/len(df1)*100)
print("\n")
display(df2.isna().sum()/len(df2)*100)

# Check for duplicates
display(df1.duplicated().sum())
display(df2.duplicated().sum())

# Check for correct data types
display(df1.dtypes)
print("\n")
display(df2.dtypes)

# Retrieve statistical information (pre-cleaning)
display(df1.describe())
display(df2.describe())

# Check if listings in categorical variable have >=10 modalities
print("Categorical variables in WHI have 10 or more modalities:\n", df1.select_dtypes(exclude="number").nunique()<10, "\n")
print("Categorical variables in WHI2021 have 10 or more modalities:\n", df2.select_dtypes(exclude="number").nunique()<10, "\n")

# Retrieve information about the categorical variables
print("Number of different countries in WHI:", df1["Country name"].value_counts().count())
print("Unique frequencies of countries listed in WHI:", df1["Country name"].value_counts().unique(),"\n")
# We can see that at least some of the countries appear only a few times

print("Number of different countries in WHI2021:", df2["Country name"].value_counts().count())
print("Number of regional indicators in WHI2021:", df2["Regional indicator"].value_counts().count())

# Preparing visualisation of distributions by normalizing (so that "year" and "Healthy life expectancy at birth" are more comparable to the other variables)
num_df1 = df1.select_dtypes("number")
norm_df1 = (num_df1-num_df1.mean())/num_df1.std()
num_df2 = df2.select_dtypes("number")
norm_df2 = (num_df2-num_df2.mean())/num_df2.std()

# Visualisation of normalized distributions for WHI and WHI2021
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
bx1 = norm_df1.boxplot(ax=ax1)
bx1.set_xticklabels(bx1.get_xticklabels(), rotation=90)
bx2 = norm_df2.boxplot(ax=ax2)
bx2.set_xticklabels(bx2.get_xticklabels(), rotation=90);

# Create a dictionary of shape {"Country name": "Regional indicator"} for assigning
df_reg = df2[["Country name", "Regional indicator"]].copy()
reg_dict = df_reg.set_index("Country name")["Regional indicator"].to_dict()

# Including and matching regional indicator to WHI dataset
df1_prep = df1.copy()
df1_prep["Regional indicator"] = df1_prep["Country name"].map(reg_dict)

# Preparing WHI 2021 dataset for the right format to include 2021 entries to WHI
df2_prep = df2.copy()
df2_prep["year"] = int(2021)
df2_prep = df2_prep.drop(["Standard error of ladder score", "upperwhisker", "lowerwhisker", "Ladder score in Dystopia", "Explained by: Log GDP per capita", "Explained by: Social support", "Explained by: Healthy life expectancy", "Explained by: Freedom to make life choices", "Explained by: Generosity", "Explained by: Perceptions of corruption", "Dystopia + residual"], axis=1)
df2_prep = df2_prep.rename(columns={"Ladder score":"Life Ladder", "Logged GDP per capita":"Log GDP per capita", "Healthy life expectancy":"Healthy life expectancy at birth"})

# Concatenating the 2021 entries to WHI dataset
df_merge = pd.concat([df1_prep, df2_prep])
df_merge = df_merge.sort_values(["Country name", "year"]).reset_index()
df_merge = df_merge.drop("index", axis=1)
df_merge.info()

# df_merge will be the central DataFrame that will be worked on from here on!

# Manually filling in missing regional indicators
missing_regs = {"Trinidad and Tobago":"Latin America and Caribbean", "Guyana":"Latin America and Caribbean", "Cuba":"Latin America and Caribbean", "Suriname":"Latin America and Caribbean", "Belize":"Latin America and Caribbean",
                "Syria":"Middle East and North Africa", "Qatar":"Middle East and North Africa", "Oman":"Middle East and North Africa",
                "Congo (Kinshasa)":"Sub-Saharan Africa", "Djibouti":"Sub-Saharan Africa", "Central African Republic":"Sub-Saharan Africa", "Angola":"Sub-Saharan Africa", "Somalia":"Sub-Saharan Africa", "Somaliland region":"Sub-Saharan Africa", "Sudan":"Sub-Saharan Africa", "South Sudan":"Sub-Saharan Africa",
                "Bhutan":"South Asia"}

# Creating complete regional dictionary
complete_regs = {**reg_dict, **missing_regs}
df_merge["Regional indicator"] = df_merge["Regional indicator"].fillna(df_merge["Country name"].map(complete_regs))

# Dropping 2005, since it only consists of 27 countries and can be considered an "outlier-year"
df_merge = df_merge[(df_merge["year"] != 2005)]
display(df_merge)

# Creating means per country and preparing to plot by regions
means_country = df_merge.groupby(["Country name"]).agg({"Life Ladder":"mean", "Log GDP per capita":"mean", "Social support":"mean", "Healthy life expectancy at birth":"mean", "Freedom to make life choices":"mean", "Generosity":"mean", "Perceptions of corruption":"mean", "Positive affect":"mean", "Negative affect":"mean"})
means_country = means_country.sort_values("Life Ladder", ascending=False)
means_country["Regional indicator"] = means_country.index.map(complete_regs)

# Creating means per region and preparing to plot by regions
means_region = df_merge
means_region["Regional indicator"] = means_region["Country name"].map(complete_regs)
means_region = means_region.groupby(["Regional indicator"]).agg({"Life Ladder":"mean", "Log GDP per capita":"mean", "Social support":"mean", "Healthy life expectancy at birth":"mean", "Freedom to make life choices":"mean", "Generosity":"mean", "Perceptions of corruption":"mean", "Positive affect":"mean", "Negative affect":"mean"})
means_region = means_region.sort_values("Life Ladder", ascending=False)

# Checking merged dataset for missing values
df_merge_na = df_merge.isna().sum()/len(df_merge)*100
display(df_merge_na)

# Execute for saving as XLSX-file:
#df_merge_na.to_excel("number_of_nas_column.xlsx")

# Show dimensions of original datasets and merged dataset
df_colrows = pd.DataFrame(data={"Number of columns":[len(df1.columns), len(df2.columns), len(df_merge.columns)], "Number of rows":[len(df1), len(df2), len(df_merge)]}, index=["WHR", "2021", "MERGE"])
display(df_colrows)

# Execute for saving as XLSX-file:
#df_colrows.to_excel("number_of_colrows.xlsx")

# Selecting numerical variables and standardizing them, in order to show a boxplot with distributions
num_df_merge = df_merge.select_dtypes("number")
norm_df_merge = (num_df_merge-num_df_merge.mean())/num_df_merge.std()

fig, ax = plt.subplots(figsize=(10,7))
plt.title("Distributions of standardized variables")
sns.boxplot(norm_df_merge, palette="flare", )
ax.set_yticks(range(-5,6))
ax.set_xticklabels(bx1.get_xticklabels(), rotation=90);

# Execute for saving boxplot with df_merge distributions:
#plt.tight_layout()
#fig.savefig("distributions_merge.png");

# Show descriptive statistics for df_merge
df_merge_descstats = df_merge.describe().drop("year", axis=1)
display(df_merge_descstats)

# Execute for saving as XLSX-file:
# df_merge_descstats.to_excel("df_merge_descriptive_stats.xlsx")

display(df_merge["Country name"].value_counts())

# Showing distributions for all variables per histogram
df_hist = df_merge.select_dtypes("number")

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20,12))

sns.histplot(x=df_hist["year"], kde=True, ax=axes[0,0])
axes[0,0].set_xlabel("year")
sns.histplot(x=df_hist["Life Ladder"], kde=True, ax=axes[0,1])
axes[0,1].set_xlabel("Life Ladder")
axes[0,1].set_ylabel("")
sns.histplot(x=df_hist["Log GDP per capita"], kde=True, ax=axes[0,2])
axes[0,2].set_xlabel("Log GDP per capita")
axes[0,2].set_ylabel("")
sns.histplot(x=df_hist["Social support"], kde=True, ax=axes[0,3])
axes[0,3].set_xlabel("Social support")
axes[0,3].set_ylabel("")
sns.histplot(x=df_hist["Healthy life expectancy at birth"], kde=True, ax=axes[0,4])
axes[0,4].set_xlabel("Healthy life expectancy at birth")
axes[0,4].set_ylabel("")
sns.histplot(x=df_hist["Freedom to make life choices"], kde=True, ax=axes[1,0])
axes[1,0].set_xlabel("Freedom to make life choices")
sns.histplot(x=df_hist["Generosity"], kde=True, ax=axes[1,1])
axes[1,1].set_xlabel("Generosity")
axes[1,1].set_ylabel("")
sns.histplot(x=df_hist["Perceptions of corruption"], kde=True, ax=axes[1,2])
axes[1,2].set_xlabel("Perceptions of corruption")
axes[1,2].set_ylabel("")
sns.histplot(x=df_hist["Positive affect"], kde=True, ax=axes[1,3])
axes[1,3].set_xlabel("Positive affect")
axes[1,3].set_ylabel("")
sns.histplot(x=df_hist["Negative affect"], kde=True, ax=axes[1,4])
axes[1,4].set_xlabel("Negative affect")
axes[1,4].set_ylabel("");

# Execute for saving correlation matrix for df_merge:
#plt.tight_layout()
#fig.savefig("hist_distributions_merge.png");

# 1. Life ladder averages by region
fig, ax = plt.subplots(figsize=(10,6))
plt.title("Average Ladder scores by region")

sns.barplot(x=means_region["Life Ladder"], y=means_region.index, palette="flare");
sns.set_style("whitegrid")
ax.bar_label(ax.containers[0], fmt="%.3f", padding=3);

# Execute for saving life ladder by region:
#plt.tight_layout()
#fig.savefig("life_ladder_region.png")

# 2. What Key Figures are correlated to Life Ladder and each other?
fig, ax = plt.subplots(figsize=(10,7))
plt.title("Correlation Matrix")

cor = df_merge.iloc[:,2:11].corr()
sns.heatmap(cor, annot=True, ax=ax, cmap="flare");

# Execute for saving correlation matrix for df_merge:
#plt.tight_layout()
#fig.savefig("correlations_merge.png");

# Assign 2020 with "pre" and 2021 with "post", assigning all prior reports with "none"
# Note: the report from 2020 was released before the first lockdowns, possible effects are therefore only visible beginning in 2021.
pre_post = df_merge
pre_post["pre_post"] = pre_post["year"].replace({2005: "none", 2006: "none", 2007: "none", 2008: "none", 2009: "none", 2010: "none", 2011: "none", 2012: "none", 2013: "none", 2014: "none", 2015: "none", 2016: "none", 2017: "none", 2018: "none", 2019: "none", 2020: "pre", 2021: "post"})
pre_post = pre_post[(pre_post["pre_post"] == "pre") | (pre_post["pre_post"] == "post")]

# Check for countries, that were available in the year pre/post covid only
exclusion = pd.DataFrame(pre_post["Country name"].value_counts() == 2)
exclusion = exclusion.loc[exclusion["count"] == True]
inclusion = pre_post.loc[pre_post["Country name"].isin(exclusion.index)]

# 3. Can we see a pre- and post-covid effect?
fig, ax = plt.subplots(figsize=(10,8))

hue_order = ["pre", "post"]
sns.barplot(x=inclusion["Life Ladder"], y=inclusion["Regional indicator"], hue=inclusion["pre_post"], hue_order=hue_order, order=means_region.index, errorbar=None, palette="flare")
sns.set_style("whitegrid")
ax.set_xlim([4,7.5])
ax.legend(loc="lower right", labels=["Report before Covid-19 outbreak", "Report after Covid-19 outbreak"])
ax.set_title("Regional Life Ladder comparison before and after Covid")
ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
ax.bar_label(ax.containers[1], fmt="%.3f", padding=3);

# Checking Life Ladder for regions before and after the outbreak
display(pre_post.groupby(["year", "Regional indicator"]).agg({"Life Ladder":"mean"}))

# Execute for saving life ladder by region pre and post covid:
#plt.tight_layout()
#fig.savefig("life_ladder_regio_covid.png");

# 4. Show Scatterplot of Life Ladder and GDP by region
fig, ax = plt.subplots(figsize=(12,8))

hue_order = ["North America and ANZ", "Western Europe", "Latin America and Caribbean", "East Asia", "Central and Eastern Europe", "Middle East and North Africa", "Southeast Asia", "Commonwealth of Independent States", "South Asia", "Sub-Saharan Africa"]
sns.scatterplot(x='Log GDP per capita', y='Life Ladder', data=df_merge, palette="tab10", hue='Regional indicator', hue_order=hue_order)
sns.set_style("whitegrid")
plt.title('Log GDP per Capita and Ladder Score by region')
plt.xlabel('Log GDP per capita')
plt.ylabel('Ladder Score');

# Execute for saving life ladder score and GDP by region:
#plt.tight_layout()
#fig.savefig("life_ladder_gdp_regio.png");

# When trying to display the evolution of Life Ladder over the years, one have to keep in mind that each year the number of available countries differs.
# In the following, a try for estimating a fairer comparison between years was examined.

# Check for countries, that were available in all last 5 or 10 years only
year10_mask = df_merge.loc[df_merge["year"] > 2010]
exclusion2 = pd.DataFrame(year10_mask["Country name"].value_counts() == 10)
exclusion2 = exclusion2.loc[exclusion2["count"] == True]
inclusion2 = year10_mask.loc[year10_mask["Country name"].isin(exclusion2.index)]
print("Countries available in all 10 last years:", inclusion2["Country name"].value_counts().count())

year5_mask = df_merge.loc[df_merge["year"] > 2015]
exclusion3 = pd.DataFrame(year5_mask["Country name"].value_counts() == 5)
exclusion3 = exclusion3.loc[exclusion3["count"] == True]
inclusion3 = year5_mask.loc[year5_mask["Country name"].isin(exclusion3.index)]
print("Countries available in all 5 last years:", inclusion3["Country name"].value_counts().count(),"\n")

# 5. Display evolution of Life Ladder over the years
average_life_ladder = df_merge.groupby("year")["Life Ladder"].mean().reset_index()

fig, ax = plt.subplots(figsize=(14,5))

sns.lineplot(data=average_life_ladder, x="year", y="Life Ladder", errorbar=None)
sns.set_style("whitegrid")
plt.title("Evolution of average Ladder score through the years")
plt.xlabel("Year")
plt.ylabel("Life Ladder");

# How many countries were available in each year?
display(df_merge["year"].value_counts())

# Execute for saving evolution of life ladder over the years:
#plt.tight_layout()
#fig.savefig("life_ladder_years.png");

# Import more relevant packages / libraries
from scipy.stats import pearsonr
import statsmodels.api

### 1. ANOVA: Regional indicator -> Life Ladder ###

# H0: The regional indicator does not have an effect on the Life Ladder score
# H1: The regional indicator does have an effect on the Life Ladder score

df_anova = df_merge.rename(columns={"Life Ladder": "Life_Ladder", "Regional indicator":"Regional_indicator", "Log GDP per capita":"Log_GDP_per_capita", "Social support":"Social_support", "Freedom to make life choices":"Freedom_to_make_life_choices"})
df_anova_cov = df_anova[(df_anova["pre_post"] != "none")]

result1 = statsmodels.formula.api.ols('Life_Ladder ~ Regional_indicator', data=df_anova).fit()
table1 = statsmodels.api.stats.anova_lm(result1)
display(table1)

# Conclusion: The p-value here is less than 5%, which is why we can reject H0 and assume H1.
# There is a significant effect of the regional indicator on the Life Ladder score.


### 2. ANOVA: Pre/Post Covid -> Life Ladder ###

# H0: Being before or after Covid does not have an effect on the Life Ladder score
# H1: Being before or after Covid does have an effect on the Life Ladder score

result2 = statsmodels.formula.api.ols('Life_Ladder ~ pre_post', data=df_anova_cov).fit()
table2 = statsmodels.api.stats.anova_lm(result2)
display(table2)

# Conclusion: The p-value here is less than 5%, which is why we can reject H0 and assume H1.
# There is a significant effect of being prior to or after Covid on the Life Ladder score!


### 3. ANOVA: Regional indicator -> Log GDP per capita ###

# H0: The regional indicator does not have an effect on logged GDP per capita
# H1: The regional indicator does have an effect on logged GDP per capita

result3 = statsmodels.formula.api.ols('Log_GDP_per_capita ~ Regional_indicator', data=df_anova).fit()
table3 = statsmodels.api.stats.anova_lm(result3)
display(table3)

# Conclusion: The p-value here is less than 5%, which is why we can reject H0 and assume H1.
# There is a significant effect of the regional indicator on the logged GDP per capita!

### 4. ANOVA: Pre/Post Covid -> Social Support ###

# H0: Being before or after Covid does not have an effect on the social support
# H1: Being before or after Covid does not have an effect on the social support

result4 = statsmodels.formula.api.ols('Social_support ~ pre_post', data=df_anova_cov).fit()
table4 = statsmodels.api.stats.anova_lm(result4)
display(table4)

# Conclusion: The p-value here is above 5%, which is why we cannot reject H0.
# There is no significant effect of being prior to or after Covid on the social support!

### 5. Pearson-Correlation: Generosity -> Life Ladder ###

# H0: Generosity and Life Ladder are correlated
# H1: Generosity and Life Ladder are not correlated

df_corr = df_merge.dropna(axis=0)

print("p-value: ", pearsonr(x = df_corr["Life Ladder"], y = df_corr["Generosity"])[1])
print("coefficient: ", pearsonr(x = df_corr["Life Ladder"], y = df_corr["Generosity"])[0])

# Conclusion: The p-value here is less than 5%, which is why we can reject H0 and assume H1.
# There is a significant  correlation between Generosity and Life Ladder score with a (weak) coefficient of 0.18!

# Import more relevant packages / libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Splitting df_merge into a training set and a test set
feats = df_merge.drop(["Country name", "year", "Life Ladder"], axis=1)
target = df_merge["Life Ladder"]

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25)

# Defining numerical and categorical features
num_train = X_train[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]]
num_test = X_test[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]]

cat_train = X_train[["Regional indicator", "pre_post"]]
cat_test = X_test[["Regional indicator", "pre_post"]]

# Showing value counts for identifying column names after OneHotEncoding
display(X_train["Regional indicator"].value_counts())
display(X_train["pre_post"].value_counts())

# Imputing missing values with SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
num_train = imputer.fit_transform(num_train)
num_test = imputer.transform(num_test)

# Normalize the numerical features
scaler = MinMaxScaler()
num_train = scaler.fit_transform(num_train)
num_test = scaler.transform(num_test)

# Encode categorical features
ohe = OneHotEncoder(drop="first", sparse_output=False)
cat_train = ohe.fit_transform(cat_train)
cat_test = ohe.transform(cat_test)

# Reunite the datasets and check for correct pre-processing
X_train = pd.DataFrame(np.concatenate([num_train, cat_train], axis=1))
X_test = pd.DataFrame(np.concatenate((num_test, cat_test), axis=1))

# Import more relevant packages / libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Preparations for using the PCA
df_pca = df_merge.drop(["Country name", "year", "Life Ladder", "Regional indicator", "pre_post"], axis=1)
# Standardize the data and impute missing values for PCA
std_scaler = StandardScaler()
z_pca = std_scaler.fit_transform(df_pca)
z_pca = imputer.fit_transform(z_pca)

# Conducting the PCA
pca = PCA()
coord_pca = pca.fit_transform(z_pca)

# Display the explained variance per factor of PCA (Scree plot)
plt.plot(range(1, len(df_pca.columns)+1), pca.explained_variance_ratio_)
plt.xlabel("Number of factors")
plt.ylabel("Eigenvalues")
plt.title("Share of explained variance per factor of PCA")
plt.show();

# Display the correlation circle
root_eigenvalues = np.sqrt(pca.explained_variance_)
corvar = np.zeros((len(df_pca.columns), len(df_pca.columns)))
for k in range(len(df_pca.columns)):
    corvar[:, k] = pca.components_[:, k] * root_eigenvalues[k]

# Delimitation 
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

# Displaying variables
for j in range(len(df_pca.columns)):
    plt.annotate(df_pca.columns[j], (corvar[j, 0], corvar[j, 1]), color="#091158")
    plt.arrow(0, 0, corvar[j, 0]*0.6, corvar[j, 1]*0.6,
              alpha=0.5, head_width=0.03, color="b")

# Adding Axis
plt.plot([-1, 1], [0, 0], color="silver", linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color="silver", linestyle='-', linewidth=1)

# Circle and labels
cercle = plt.Circle((0, 0), 1, color="#16E4CA", fill=False)
axes.add_artist(cercle)
plt.xlabel("AXIS 1")
plt.ylabel("AXIS 2")
plt.show();

# Import more relevant packages / libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Conduct first linear regression modelling
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Saving regression predictions on test set
pred_test = regressor.predict(X_test)

# Show coefficient of determination for train and test data
print("R² for train data:", regressor.score(X_train, y_train))
print("R² for test data:", regressor.score(X_test, y_test))

# Show performance metrics MAE, MSE, RMSE
mse = mean_squared_error(y_test, pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred_test)
print("Mean Squared Error:", mse, "\nRoot Mean Squared Error:", rmse, "\nMean Absolute Error:", mae)

# R² shows an acceptable representation of .78 (training set) and .80 (test set). We do not see indications of overfitting.

# Display the intercept and estimated coefficients of each variable
coeffs = list(regressor.coef_)
coeffs.insert(0, regressor.intercept_)

reg_feats = ["Intercept", "Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect",
             "Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"]

reg_coefficients = pd.DataFrame({'Estimated value': coeffs}, index=reg_feats)
display(reg_coefficients)

# Comparing model predictions with real target values
fig, ax = plt.subplots(figsize=(9,7))
sns.set_style("darkgrid")
plt.title("Life Ladder linear regression predictions vs. actual target values")
plt.xlabel("Modelled Life Ladder (test set)")
plt.ylabel("Actual Life Ladder (test set)")
sns.scatterplot(x=pred_test, y=y_test)
sns.lineplot(x=(y_test.min(), y_test.max()), y=(y_test.min(), y_test.max()), color="r");

# Execute for saving life ladder predictions versus actual values:
#plt.tight_layout()
#fig.savefig("scatterplot_reg_predicted_01.png");

# Import more relevant packages / libraries
from sklearn.tree import DecisionTreeRegressor

# Conduct first decision tree regression modelling
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Saving decision tree regression predictions on test set
y_pred_dt = dt_regressor.predict(X_test)

# Show coefficient of determination for train and test data
print("R² for train data:", dt_regressor.score(X_train, y_train))
print("R² for test data:", dt_regressor.score(X_test, y_test))

# Show performance metrics MAE, MSE, RMSE
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
print("Mean Squared Error:", mse_dt, "\nRoot Mean Squared Error:", rmse_dt, "\nMean Absolute Error:", mae_dt)

# Display the estimated importance of each feature
dt_feat_importances = pd.DataFrame(dt_regressor.feature_importances_, columns=["Importance"])
dt_feat_importances.rename(index={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"}, inplace=True)
dt_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)
display(dt_feat_importances)

# Comparing model predictions with real target values
fig, ax = plt.subplots(figsize=(9,7))
sns.set_style("darkgrid")
plt.title("Life Ladder decision tree regression predictions vs. actual target values")
plt.xlabel("Modelled Life Ladder (test set)")
plt.ylabel("Actual Life Ladder (test set)")
sns.scatterplot(x=y_pred_dt, y=y_test)
sns.lineplot(x=(y_test.min(), y_test.max()), y=(y_test.min(), y_test.max()), color="r");

# Execute for saving life ladder predictions versus actual values:
#plt.tight_layout()
#fig.savefig("scatterplot_dtreg_predicted_01.png");

# Import more relevant packages / libraries
from sklearn.ensemble import RandomForestRegressor

# Conduct first random forest regression modelling
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42)
rf_regressor.fit(X_train, y_train)

# Saving random forest regression predictions on test set
y_pred_rf = rf_regressor.predict(X_test)

# Show coefficient of determination for train and test data
print("R² for train data:", rf_regressor.score(X_train, y_train))
print("R² for test data:", rf_regressor.score(X_test, y_test))

# Show performance metrics MAE, MSE, RMSE
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("Mean Squared Error:", mse_rf, "\nRoot Mean Squared Error:", rmse_rf, "\nMean Absolute Error:", mae_rf)

# Display the estimated importance of each feature
rf_feat_importances = pd.DataFrame(rf_regressor.feature_importances_, columns=["Importance"])
rf_feat_importances.rename(index={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"}, inplace=True)
rf_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)
display(rf_feat_importances)

# Comparing model predictions with real target values
fig, ax = plt.subplots(figsize=(9,7))
sns.set_style("darkgrid")
plt.title("Life Ladder random forest regression predictions vs. actual target values")
plt.xlabel("Modelled Life Ladder (test set)")
plt.ylabel("Actual Life Ladder (test set)")
sns.scatterplot(x=y_pred_rf, y=y_test)
sns.lineplot(x=(y_test.min(), y_test.max()), y=(y_test.min(), y_test.max()), color="r");

# Execute for saving life ladder predictions versus actual values:
#plt.tight_layout()
#fig.savefig("scatterplot_rfreg_predicted_01.png");

# Import more relevant packages / libraries
from xgboost import XGBRegressor

# Conduct first XGBoost regression modelling
xgb_reg = XGBRegressor(random_state=42)
xgb_reg.fit(X_train, y_train)

# Saving XGBoost regression predictions on test set
y_pred_xgb = xgb_reg.predict(X_test)

# Show coefficient of determination for train and test data
print("R² for train data:", xgb_reg.score(X_train, y_train))
print("R² for test data:", xgb_reg.score(X_test, y_test))

# Show performance metrics MAE, MSE, RMSE
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print("Mean Squared Error:", mse_xgb, "\nRoot Mean Squared Error:", rmse_xgb, "\nMean Absolute Error:", mae_xgb)

# Display the estimated importance for each feature
xgb_feat_importances = pd.DataFrame(xgb_reg.feature_importances_, columns=["Importance"])
xgb_feat_importances.rename(index={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"}, inplace=True)
xgb_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)
display(xgb_feat_importances)

# Comparing model predictions with real target values
fig, ax = plt.subplots(figsize=(9,7))
sns.set_style("darkgrid")
plt.title("Life Ladder xgb regression predictions vs. actual target values")
plt.xlabel("Modelled Life Ladder (test set)")
plt.ylabel("Actual Life Ladder (test set)")
sns.scatterplot(x=y_pred_xgb, y=y_test)
sns.lineplot(x=(y_test.min(), y_test.max()), y=(y_test.min(), y_test.max()), color="r");

# Execute for saving life ladder predictions versus actual values:
#plt.tight_layout()
#fig.savefig("scatterplot_xgbreg_predicted_01.png");

# Import more relevant packages / libraries
import shap

# Use SHAP technique to further investigate the feature importance for the regression
explainer_regressor = shap.LinearExplainer(regressor, X_train)
shap_values_regressor = explainer_regressor.shap_values(X_train)

sample_idx = 0
shap.initjs()
shap.force_plot(explainer_regressor.expected_value, shap_values_regressor[sample_idx, :], X_train.iloc[sample_idx, :])

# Visualize the shap scores results
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values_regressor, X_train, cmap="flare");

# Execute for saving shap values per feature:
#plt.tight_layout()
#fig.savefig("shap_values_regressor_01.png");

# Removing variables with a coefficient of under 0.1
X_train_2 = pd.DataFrame(X_train).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_train_2 = X_train_2.drop(["Commonwealth of Independent States", "Middle East and North Africa", "post", "pre"], axis=1)

X_test_2 = pd.DataFrame(X_test).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_test_2 = X_test_2.drop(["Commonwealth of Independent States", "Middle East and North Africa", "post", "pre"], axis=1)

# Conducting second linear regression with less variables
regressor2 = LinearRegression()
regressor2.fit(X_train_2, y_train)

# Saving regression predictions on test set
pred_test2 = regressor2.predict(X_test_2)

# Show coefficient of determination for train and test data
print("R² for train data:", regressor2.score(X_train_2, y_train))
print("R² for test data:", regressor2.score(X_test_2, y_test))

# Show performance metrics MAE, MSE, RMSE (optional, since we are dealing with a linear regression model)
mse2 = mean_squared_error(y_test, pred_test2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_test, pred_test2)
print("Mean Squared Error:", mse2, "\nRoot Mean Squared Error:", rmse2, "\nMean Absolute Error:", mae2)

# Display the intercept and estimated coefficients of each variable
coeffs2 = list(regressor2.coef_)
coeffs2.insert(0, regressor2.intercept_)

reg_feats2 = ["Intercept", "Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect",
             "East Asia", "Latin America and Caribbean", "North America and ANZ", "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe"]

reg_coefficients2 = pd.DataFrame({"Estimated value": coeffs2}, index=reg_feats2)
display(reg_coefficients2)

# Removing pre/post and all regional indicators
X_train_3 = pd.DataFrame(X_train).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_train_3 = X_train_3.drop(["Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

X_test_3 = pd.DataFrame(X_test).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_test_3 = X_test_3.drop(["Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

# Conducting third linear regression with less variables
regressor3 = LinearRegression()
regressor3.fit(X_train_3, y_train)

# Saving regression predictions on test set
pred_test3 = regressor3.predict(X_test_3)

# Show coefficient of determination for train and test data
print("R² for train data:", regressor3.score(X_train_3, y_train))
print("R² for test data:", regressor3.score(X_test_3, y_test))

# Show performance metrics MAE, MSE, RMSE (optional, since we are dealing with a linear regression model)
mse3 = mean_squared_error(y_test, pred_test3)
rmse3 = np.sqrt(mse3)
mae3 = mean_absolute_error(y_test, pred_test3)
print("Mean Squared Error:", mse3, "\nRoot Mean Squared Error:", rmse3, "\nMean Absolute Error:", mae3)

# Display the intercept and estimated coefficients of each variable
coeffs3 = list(regressor3.coef_)
coeffs3.insert(0, regressor3.intercept_)

reg_feats3 = ["Intercept", "Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]

reg_coefficients3 = pd.DataFrame({"Estimated value": coeffs3}, index=reg_feats3)
display(reg_coefficients3)

# Removing pre/post, all regional indicators, freedom, generosity, corruption and negative affect
X_train_4 = pd.DataFrame(X_train).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_train_4 = X_train_4.drop(["Freedom to make life choices", "Generosity", "Perceptions of corruption", "Negative affect", "Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", 
                            "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

X_test_4 = pd.DataFrame(X_test).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_test_4 = X_test_4.drop(["Freedom to make life choices", "Generosity", "Perceptions of corruption", "Negative affect", "Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", 
                          "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

# Conducting fourth linear regression with less variables
regressor4 = LinearRegression()
regressor4.fit(X_train_4, y_train)

# Saving regression predictions on test set
pred_test4 = regressor4.predict(X_test_4)

# Show coefficient of determination for train and test data
print("R² for train data:", regressor4.score(X_train_4, y_train))
print("R² for test data:", regressor4.score(X_test_4, y_test))

# Show performance metrics MAE, MSE, RMSE (optional, since we are dealing with a linear regression model)
mse4 = mean_squared_error(y_test, pred_test4)
rmse4 = np.sqrt(mse4)
mae4 = mean_absolute_error(y_test, pred_test4)
print("Mean Squared Error:", mse4, "\nRoot Mean Squared Error:", rmse4, "\nMean Absolute Error:", mae4)

# Display the intercept and estimated coefficients of each variable
coeffs4 = list(regressor4.coef_)
coeffs4.insert(0, regressor4.intercept_)

reg_feats4 = ["Intercept", "Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Positive affect"]

reg_coefficients4 = pd.DataFrame({"Estimated value": coeffs4}, index=reg_feats4)
display(reg_coefficients4)

# Import more relevant packages / libraries
from sklearn.model_selection import GridSearchCV

# Conduct GridSearchCV in order to find best given variants of hyperparameters for the model
param_grid = {
    "n_estimators": [70, 80, 90],  
    "min_samples_split": [2, 3, 4]
}

rf = RandomForestRegressor(random_state=42)
    
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="r2")
grid_search.fit(X_train, y_train)

# Saving the best found parameters and estimators as objects
best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_

# Printing them as results
print("Best Parameters:", best_parameters)
print("R² Score (training):", best_model.score(X_train, y_train))
print("R² Score (test):", best_model.score(X_test, y_test))

# Use SHAP technique to further investigate the feature importance for the decision tree
explainer_rf = shap.TreeExplainer(rf_regressor)
shap_values_rf = explainer_rf.shap_values(X_train)

sample_idx = 0
shap.initjs()
shap.force_plot(explainer_rf.expected_value, shap_values_rf[sample_idx, :], X_train.iloc[sample_idx, :])

# Visualize the shap scores results
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values_rf, X_train, cmap="flare");

# Execute for saving shap values per feature:
#plt.tight_layout()
#fig.savefig("shap_values_rf_regressor_01.png");

# Conduct second random forest regression modeling
rf_regressor2 = RandomForestRegressor(
    n_estimators=80,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=5,
    random_state=42)
rf_regressor2.fit(X_train, y_train)

# Saving random forest regression predictions on test set
y_pred_rf2 = rf_regressor2.predict(X_test)

# Show coefficient of determination for train and test data
print("R² for train data:", rf_regressor2.score(X_train, y_train))
print("R² for test data:", rf_regressor2.score(X_test, y_test))

# Show performance metrics MAE, MSE, RMSE
mse_rf2 = mean_squared_error(y_test, y_pred_rf2)
rmse_rf2 = np.sqrt(mse_rf2)
mae_rf2 = mean_absolute_error(y_test, y_pred_rf2)
print("Mean Squared Error:", mse_rf2, "\nRoot Mean Squared Error:", rmse_rf2, "\nMean Absolute Error:", mae_rf2)

# Display the estimated importance of each feature
rf_feat_importances2 = pd.DataFrame(rf_regressor2.feature_importances_, columns=["Importance"])
rf_feat_importances2.rename(index={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"}, inplace=True)
rf_feat_importances2.sort_values(by="Importance", ascending=False, inplace=True)
display(rf_feat_importances2)

# Removing pre/post and the 6 least important regions 
X_train_5 = pd.DataFrame(X_train).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_train_5 = X_train_5.drop(["Commonwealth of Independent States" , "Middle East and North Africa", "North America and ANZ", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)
X_test_5 = pd.DataFrame(X_test).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
X_test_5 = X_test_5.drop(["Commonwealth of Independent States" , "Middle East and North Africa", "North America and ANZ", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

# Conduct third random forest regression modeling
rf_regressor3 = RandomForestRegressor(
    n_estimators=80,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=5,
    random_state=42)
rf_regressor3.fit(X_train_5, y_train)

# Saving random forest regression predictions on test set
y_pred_rf3 = rf_regressor3.predict(X_test_5)

# Show coefficient of determination for train and test data
print("R² for train data:", rf_regressor3.score(X_train_5, y_train))
print("R² for test data:", rf_regressor3.score(X_test_5, y_test))

# Show performance metrics MAE, MSE, RMSE
mse_rf3 = mean_squared_error(y_test, y_pred_rf3)
rmse_rf3 = np.sqrt(mse_rf3)
mae_rf3 = mean_absolute_error(y_test, y_pred_rf3)
print("Mean Squared Error:", mse_rf3, "\nRoot Mean Squared Error:", rmse_rf3, "\nMean Absolute Error:", mae_rf3)

# Display the estimated importance of each feature
rf_feat_importances3 = pd.DataFrame(rf_regressor3.feature_importances_, columns=["Importance"])
rf_feat_importances3.rename(index={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"East Asia", 9:"Latin America and Caribbean", 10:"South Asia"}, inplace=True)
rf_feat_importances3.sort_values(by="Importance", ascending=False, inplace=True)
display(rf_feat_importances3)

# X_train_3 and X_test_3 fulfill exactly this masking

# Conduct fourth random forest regression modeling
rf_regressor4 = RandomForestRegressor(
    n_estimators=80,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=5,
    random_state=42)
rf_regressor4.fit(X_train_3, y_train)

# Saving random forest regression predictions on test set
y_pred_rf4 = rf_regressor4.predict(X_test_3)
# Show coefficient of determination for train and test data
print("R² for train data:", rf_regressor4.score(X_train_3, y_train))
print("R² for test data:", rf_regressor4.score(X_test_3, y_test))

# Show performance metrics MAE, MSE, RMSE
mse_rf4 = mean_squared_error(y_test, y_pred_rf4)
rmse_rf4 = np.sqrt(mse_rf4)
mae_rf4 = mean_absolute_error(y_test, y_pred_rf4)
print("Mean Squared Error:", mse_rf4, "\nRoot Mean Squared Error:", rmse_rf4, "\nMean Absolute Error:", mae_rf4)

# Display the estimated importance of each feature
rf_feat_importances4 = pd.DataFrame(rf_regressor4.feature_importances_, columns=["Importance"])
rf_feat_importances4.rename(index={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect"}, inplace=True)
rf_feat_importances4.sort_values(by="Importance", ascending=False, inplace=True)
display(rf_feat_importances4)