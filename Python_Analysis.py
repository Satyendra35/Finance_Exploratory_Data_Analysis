#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#Import application.csv data.

app0=pd.read_csv('application_data.csv')
app0.head()


# In[14]:


#Need to increase the the visibility of row and columns for more clearity.

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

app0.head()


# In[7]:


#Import Previous application.csv data also

pre0=pd.read_csv('previous_application.csv')


# In[15]:


pre0.head()


# In[17]:


#Understanding the data for mor informations:-

app0.info()


# In[18]:


pre0.info()


# In[19]:


app0.describe().T


# In[20]:


#Now need to remove some unecessory columns from here:-

list(app0.columns)


# In[35]:


app0 = app0.filter(['SK_ID_CURR',
 'TARGET',
 'NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'REGION_POPULATION_RELATIVE',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'OCCUPATION_TYPE',
 'CNT_FAM_MEMBERS',
 'FLAG_MOBIL',
 'FLAG_EMP_PHONE',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_EMAIL',                                         
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'WEEKDAY_APPR_PROCESS_START',
 'HOUR_APPR_PROCESS_START',
 'REG_REGION_NOT_LIVE_REGION',
 'REG_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'ORGANIZATION_TYPE',
 'EXT_SOURCE_1',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3'])


# In[36]:


app0.info()


# In[37]:


app0.shape


# In[38]:


#Now to find the percentage of null values:-

(app0.isnull().sum() * 100 / len(app0)).round(2)


# In[39]:


#Now to calculate percentage of NaN values:-
def get_perc_of_missing_values(series):
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 3)
get_perc_of_missing_values(app0)


# In[40]:


# Iterate over columns in DataFrame and delete those with where >30% of the values are null
for col, values in app0.iteritems():
    if get_perc_of_missing_values(app0[col]) > 0.30:
        app0.drop(col, axis=1, inplace=True)
app0


# In[41]:


app0.shape


# In[42]:


#Again find the % of null values in each column, to determine what needs to be done as part of clean
(app0.isnull().sum() * 100 / len(app0)).round(2)


# In[43]:


# External_Source_3 Column has almost 20% missing value hence need to drop it:-

app0.drop("EXT_SOURCE_3",axis=1,inplace=True)


# In[44]:


app0.describe().T


# In[45]:


for col, values in app0.iteritems():
    if get_perc_of_missing_values(app0[col]) < 0.01:
        print(col,get_perc_of_missing_values(app0[col]))


# In[46]:


# Since from the above code we can intepret that "AMT_GOODS_PRICE" ,"EXT_SOURCE_2," has very low missing values.
# Hence we can fill those columns with Mean Values

# Beacuse for the other columns CNT_SOCIAL_CIRCLE series there is no definite mean , percentile values,SO that we can fill will
#mean values

app0['AMT_GOODS_PRICE'].fillna((app0['AMT_GOODS_PRICE'].mean()), inplace=True)

app0['EXT_SOURCE_2'].fillna((app0['EXT_SOURCE_2'].mean()), inplace=True)


# In[47]:


# Mean values are filled

app0[["AMT_GOODS_PRICE","EXT_SOURCE_2"]].describe().T


# In[48]:


app0.NAME_TYPE_SUITE.value_counts()


# In[49]:


#Here "Unaccompanied" data has the highest mode.We can fill missing values  with Unaccompanied

app0["NAME_TYPE_SUITE"].fillna(app0["NAME_TYPE_SUITE"].mode()[0],inplace=True)


# In[50]:


## Now finding Outliers:-

app0.describe().T


# In[51]:


## As AMT_ANNUITY,CNT_FAM_MEMBERS columns has very,very few NaN values, we can fill it with mean value:-

app0['AMT_ANNUITY'].fillna((app0['AMT_ANNUITY'].mean()), inplace=True)
app0['CNT_FAM_MEMBERS'].fillna((app0['CNT_FAM_MEMBERS'].mean()), inplace=True)


# In[52]:


app0.describe().T


# In[53]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
outlier_df = app0.select_dtypes(include=numerics)


# In[54]:


outlier_df.shape


# In[55]:


outlier_df.describe().T


# In[58]:


for cols,values in outlier_df.iteritems():
    f, ax = plt.subplots(1,1,figsize=(8,8))
    f.suptitle(cols, fontsize=16)
    outlier_plot = sns.distplot(outlier_df[cols]) 


# In[59]:


# Above box plot for CNT_CHILDREN shows a large outlier(19). SInce a family cannot or very rare to have 19 children.

# In the DAYS_EMPLOYED there is a value present at 36k range, this wont be possible.This could have occured during data entry

# In the plot AMT_INCOME_TOTAL, we can vicually see that the MAX amount is way largert than the other statistical datas[Mean,(25,50,75)percentiles]


# In[60]:


app0[["CNT_CHILDREN","DAYS_EMPLOYED","AMT_INCOME_TOTAL"]].describe().T


# In[61]:


q1=app0["CNT_CHILDREN"].quantile(0.99)
q1


# In[62]:


app0.CNT_CHILDREN.value_counts()


# In[63]:


q1=app0["CNT_CHILDREN"].quantile(0.99)
app0["CNT_CHILDREN"] =  app0.CNT_CHILDREN.apply(lambda x: q1 if x>q1 else x)

f, ax = plt.subplots(figsize=(10,6))
outlier_plot_1 = sns.distplot(app0["CNT_CHILDREN"])


# In[64]:


q2=app0["DAYS_EMPLOYED"].quantile(0.80)
app0["DAYS_EMPLOYED"] =  app0.DAYS_EMPLOYED.apply(lambda x: q2 if x>q2 else x)

f, ax = plt.subplots(figsize=(10,6))
outlier_plot_2 = sns.distplot(app0["DAYS_EMPLOYED"])


# In[65]:


q3=app0["AMT_INCOME_TOTAL"].quantile(0.95)
app0["AMT_INCOME_TOTAL"] =  app0.AMT_INCOME_TOTAL.apply(lambda x: q3 if x>q3 else x)

f, ax = plt.subplots(figsize=(10,6))
outlier_plot_3 = sns.distplot(app0["AMT_INCOME_TOTAL"])


# In[66]:


app0.head(6)


# In[67]:


# Converting DAYS_BIRTH to AGE:-

app0["AGE"] = app0.DAYS_BIRTH.apply(lambda x :round(abs(x)/365),0)
app0["AGE"]
app0["AGE"] = pd.to_numeric(app0["AGE"])


# In[68]:


# Dropping "DAYS_BIRTH" column, since we have converted the DAYS to AGE:-

app0.drop("DAYS_BIRTH",axis=1,inplace=True)


# In[69]:


app0.AMT_INCOME_TOTAL.quantile([0.25,0.5,0.85,1])


# In[70]:


app0.CNT_FAM_MEMBERS.unique()


# In[71]:


#Since 2.15 member can be present, we are rounding off the values.:-

app0.CNT_FAM_MEMBERS = app0.CNT_FAM_MEMBERS.apply(lambda x : round(x,0))


# # Binning Salary Amount to Categories for more Clarity

# In[72]:


# Based on the Quantile Values , segregating the values to its respective categories:-

def salary_category_func(x):
    if x<337500 and x>=234000:
        return('HIGH')
    elif x<234000 and x>=147150:
        return('MODERATE')
    elif x<147150 and x>=112500:
        return('LOW')
    else:
        return("EXTREMLY LOW")
app0["SALARY_CATEGORY"] = app0.AMT_INCOME_TOTAL.apply(salary_category_func)    
app0["SALARY_CATEGORY"]


# In[73]:


app0.SALARY_CATEGORY.value_counts()


# In[74]:


#Dropping "AMT_INCOME_TOTAL" Column,because have Binned those Salary Values:-

app0.drop("AMT_INCOME_TOTAL",axis=1,inplace=True)


# # Analysing the count of Target variables

# In[75]:


Target_count= sns.countplot("TARGET",data =app0)

# We can see there is a huge imbalance in out Target variable. So we can segregate the Target variable into two different dataframes
# # Dividing the Application into Two Dataframes based on the Target Variable

# In[77]:


good_client = app0[app0.TARGET == 0]
defaulter_client = app0[app0.TARGET == 1]
good_client.info()


# In[78]:


defaulter_client.info()


# # Check for clients who are unlikely to pay the loans

# In[79]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Univariate_defaulter_Num_1_df = defaulter_client.select_dtypes(include=numerics)
Univariate_defaulter_Num_1_df


# In[80]:


categorical = ["object"]
Univariate_defaulter_Cat_1_df =defaulter_client.select_dtypes(include=categorical)
Univariate_defaulter_Cat_1_df


# # FLAG_OWN_REALTY

# In[81]:


#Let's compare and check how the possesion of a property affects the repayment of loans.


# In[82]:


#graph to plot Number of property owners and non-property owners in the entire polulation:-

PropertyOwners_vs_Total = sns.countplot("FLAG_OWN_REALTY",data =app0)

From above plot we can see that clients having some sort of real estate acquirements are more that the ones who don't have any property.
# In[83]:


#find the percentage of property and non-property owners in the defaulter list:-

test_df1=round((Univariate_defaulter_Cat_1_df["FLAG_OWN_REALTY"].value_counts()/app0["FLAG_OWN_REALTY"].value_counts())*100,2)
test_df1 = pd.DataFrame(test_df1)
test_df1.reset_index(level=0, inplace=True)
test_df1.rename(columns=  {"index": "FLAG_OWN_REALTY", 
                     "FLAG_OWN_REALTY":"Default_Percentage"}, 
                                 inplace = True) 
test_df1


# In[84]:


#plot to show the number of property and non-property owners vs. Target variable. Here, Target =1:-

PropertyOwners_vs_Target= sns.barplot(x="FLAG_OWN_REALTY",y="Default_Percentage",data=test_df1)

From above graph, we can see that the number of non-payers of loan i.e., defaulters are very close almost equal to 9%. It is difficult to decide a target based on this metric.
# # CODE_GENDER
Let's compare and check how the gender of client affects the repayment of loans.
# In[85]:


#graph to plot Males and femals in the entire polulation:-

Gender_vs_Total = sns.countplot("CODE_GENDER",data =app0)


# In[86]:


#find the percentage of males and females in the defaulter list:-

test_df2=round((Univariate_defaulter_Cat_1_df["CODE_GENDER"].value_counts()/app0["CODE_GENDER"].value_counts())*100,2)
test_df2 = pd.DataFrame(test_df2)
test_df2.reset_index(level=0, inplace=True)
test_df2.rename(columns=  {"index": "CODE_GENDER", 
                     "CODE_GENDER":"Default_Percentage"}, 
                                 inplace = True) 
test_df2


# In[87]:


#plot to show the number male and female clients vs. Target variable. Here, Target =1:-

Gender_vs_Target= sns.barplot(x="CODE_GENDER",y="Default_Percentage",data=test_df2)

So, from above plots and data we can cleary see that the Female clients are a better TARGET as compared to the Male clients. Observing the percent of defaulted credits, male client have a higher chance of not returning their loans [10.14%], compared to the female clients [7%].
# # _FLAG_OWN_CAR _
Let's compare and check how the car owners and non-car owners differ in their repayment of loans.
# In[88]:


#graph to plot car owners and non-carOwners in the entire polulation:-

CarOwner_vs_Total = sns.countplot("FLAG_OWN_CAR",data =app0)

Clients who don't own cars/vehicles are more in the given polultaion.
# In[89]:


#find the percentage of car owners and non-carOwners in the defaulter list:-

test_df3=round((Univariate_defaulter_Cat_1_df["FLAG_OWN_CAR"].value_counts()/app0["FLAG_OWN_CAR"].value_counts())*100,2)
test_df3 = pd.DataFrame(test_df3)
test_df3.reset_index(level=0, inplace=True)
test_df3.rename(columns=  {"index": "FLAG_OWN_CAR", 
                     "FLAG_OWN_CAR":"Default_Percentage"}, 
                                 inplace = True) 
test_df3


# In[90]:


#plot to show the car owners and non-carOwners vs. Target variable. Here, Target =1:-

CarOwner_vs_Target= sns.barplot(x="FLAG_OWN_CAR",y="Default_Percentage",data=test_df3)

As we can see from above graph,the clients that own a car are less likely to not repay the loan when compared to the ones that do not own a car. The loan non-repayment rates of both the Car Owners and Non-Car Owners are very close. Which is interesting to see and indicates that probably this metric will not be a suitable one when targeting a client.
# # NAME_FAMILY_STATUS
Let's compare and check how the family status of clients affect their repayment of loans.
# In[91]:


x = app0.NAME_FAMILY_STATUS.value_counts()
x = pd.DataFrame(x)
x.reset_index(level=0, inplace=True)
x.rename(columns=  {"index": "NAME_FAMILY_STATUS", 
                     "NAME_FAMILY_STATUS":"number"}, 
                                 inplace = True) 
x


# In[92]:


#graph to plot family status of clients in the entire polulation:-

FamilyStatus_vs_Total = sns.barplot(x="NAME_FAMILY_STATUS",y="number",data =x)
FamilyStatus_vs_Total.set_xticklabels(FamilyStatus_vs_Total.get_xticklabels(),rotation=90)

Most of clients are married, followed by Single and civil marriage.
# In[93]:


#find the percentage of clients according to family status in the defaulter list:-

test_df4=round((Univariate_defaulter_Cat_1_df["NAME_FAMILY_STATUS"].value_counts()/app0["NAME_FAMILY_STATUS"].value_counts())*100,2)
test_df4 = pd.DataFrame(test_df4)
test_df4.reset_index(level=0, inplace=True)
test_df4.rename(columns=  {"index": "NAME_FAMILY_STATUS", 
                     "NAME_FAMILY_STATUS":"Default_Percentage"}, 
                                 inplace = True) 
test_df4.sort_values(by='Default_Percentage', inplace=True)
test_df4


# In[94]:


#plot to show the family status of client vs. Target variable. Here, Target =1:-

FamilyStatus_vs_Target= sns.barplot(x="NAME_FAMILY_STATUS",y="Default_Percentage",data=test_df4)
FamilyStatus_vs_Target.set_xticklabels(FamilyStatus_vs_Target.get_xticklabels(),rotation=90)

From above graph we can say that the percentage of non-repayment of loan is at highest for civil mariage and is lowest for widows. Which is interesting to see because you expect widows to not payback their loans but it is the opposite here.
# # CNT_CHILDREN
Now, let's compare how the number of children in a family affects the non-repayment of loans.
# In[95]:


#graph to plot Number of children per client in the entire polulation:-

NoOfChildren_vs_Total = sns.countplot("CNT_CHILDREN",data =app0)


# In[96]:


#find the percentage  number of children per client the defaulter list:-

test_df5=round((Univariate_defaulter_Num_1_df["CNT_CHILDREN"].value_counts()/app0["CNT_CHILDREN"].value_counts())*100,2)
test_df5 = pd.DataFrame(test_df5)
test_df5.reset_index(level=0, inplace=True)

test_df5.rename(columns=  {"index": "CNT_CHILDREN", 
                     "CNT_CHILDREN":"Default_Percentage"}, 
                                 inplace = True)
test_df5.sort_values(by=["Default_Percentage"],ascending=False,inplace=True)
test_df5


# In[97]:


#plot to show the number of children per client vs. Target variable. Here, Target =1

#NoOfChildren_vs_Target= sns.barplot(x="CNT_CHILDREN",y="Default_Percentage",data=test_df5,order=test_df5['CNT_CHILDREN'])

NoOfChildren_vs_Target = sns.distplot(test_df5["Default_Percentage"])

Ther is more chance for a client with more children to not repay the loan back. This can be beacuse of the more liability that is on the client. The more the number of children the more difficult it is for the client to repay the loan due to more personal expenditures.
# # CNT_FAM_MEMBERS

# In[98]:


y = app0.CNT_FAM_MEMBERS.value_counts()
y = pd.DataFrame(y)
y.reset_index(level=0, inplace=True)
y.rename(columns=  {"index": "CNT_FAM_MEMBERS", 
                     "CNT_FAM_MEMBERS":"number"}, 
                                 inplace = True) 
y


# In[99]:


#graph to plot Number of family members per client in the entire polulation:-

NoOfFamilyMembers_vs_Total = sns.barplot(x="CNT_FAM_MEMBERS",y="number",data =y)
NoOfFamilyMembers_vs_Total.set_xticklabels(NoOfFamilyMembers_vs_Total.get_xticklabels(),rotation=90)


# In[100]:


#find the percentage  number of family members per client the defaulter list:-

test_df6=round((Univariate_defaulter_Num_1_df["CNT_FAM_MEMBERS"].value_counts()/app0["CNT_FAM_MEMBERS"].value_counts())*100,2)
test_df6 = pd.DataFrame(test_df6)
test_df6.reset_index(level=0, inplace=True)
test_df6.rename(columns=  {"index": "CNT_FAM_MEMBERS", 
                     "CNT_FAM_MEMBERS":"Default_Percentage"}, 
                                 inplace = True) 
test_df6.sort_values(by=["Default_Percentage"],ascending=False,inplace=True)
test_df6


# In[101]:


#plot to show the number of family members per client vs. Target variable. Here, Target =1:-

NoOfFamilyMembers_vs_Target= sns.barplot(x="CNT_FAM_MEMBERS",y="Default_Percentage",data=test_df6)
NoOfFamilyMembers_vs_Target.set_xticklabels(NoOfFamilyMembers_vs_Target.get_xticklabels(),rotation=90)

#NoOfFamilyMembers_vs_Target = sns.distplot(y["number"])

Though we can see that family with 11,13 members shows highest default rate, but their count is very less[2].
# # NAME_EDUCATION_TYPE

# In[102]:


#graph to plot education type in the entire polulation:-

EducationType_vs_Total = sns.countplot("NAME_EDUCATION_TYPE",data =app0)
EducationType_vs_Total.set_xticklabels(EducationType_vs_Total.get_xticklabels(),rotation=90)


# In[103]:


#find the percentage education level of clients in the defaulter list:-

test_df7=round((Univariate_defaulter_Cat_1_df["NAME_EDUCATION_TYPE"].value_counts()/app0["NAME_EDUCATION_TYPE"].value_counts())*100,2)

test_df7 = pd.DataFrame(test_df7)

test_df7.reset_index(level=0, inplace=True)
test_df7.sort_values(by=["NAME_EDUCATION_TYPE"],ascending=False,inplace=True)

test_df7.rename(columns=  {"index": "NAME_EDUCATION_TYPE", 
                     "NAME_EDUCATION_TYPE":"Default_Percentage"}, 
                                  inplace = True) 


test_df7


# In[104]:


#plot to show the education type of each client vs. Target variable. Here, Target =1:-

f, ax = plt.subplots(figsize=(8,6))
EducationType_vs_Target= sns.barplot(x="NAME_EDUCATION_TYPE",y="Default_Percentage",data=test_df7,order=test_df7['NAME_EDUCATION_TYPE'])
EducationType_vs_Target.set_xticklabels(EducationType_vs_Target.get_xticklabels(), rotation=45)
EducationType_vs_Target

It can be seen from above graph that the more educated clients are likely to repay their loans because they will be having more stable jobs with monthly income.
# # _NAME_TYPE_SUITE

# In[105]:


TypeSuite_vs_Total = sns.countplot("NAME_TYPE_SUITE",data =app0)
TypeSuite_vs_Total.set_xticklabels(TypeSuite_vs_Total.get_xticklabels(),rotation=90)


# In[106]:


test_df8=round((Univariate_defaulter_Cat_1_df["NAME_TYPE_SUITE"].value_counts()/app0["NAME_TYPE_SUITE"].value_counts())*100,2)

test_df8 = pd.DataFrame(test_df8)

test_df8.reset_index(level=0, inplace=True)
test_df8.sort_values(by=["NAME_TYPE_SUITE"],ascending=False,inplace=True)

test_df8.rename(columns=  {"index": "NAME_TYPE_SUITE", 
                     "NAME_TYPE_SUITE":"Default_Percentage"}, 
                                  inplace = True) 


test_df8


# In[107]:


f, ax = plt.subplots(figsize=(8,6))
TypeSuite_vs_Target= sns.barplot(x="NAME_TYPE_SUITE",y="Default_Percentage",data=test_df8
                                         ,order=test_df8['NAME_TYPE_SUITE'])
TypeSuite_vs_Target.set_xticklabels(TypeSuite_vs_Target.get_xticklabels(), rotation=45)
TypeSuite_vs_Target


# #ORGANISATION_TYPE

# In[108]:


test_df9=round((Univariate_defaulter_Cat_1_df["ORGANIZATION_TYPE"].value_counts()/app0["ORGANIZATION_TYPE"].value_counts())*100,2)

test_df9 = pd.DataFrame(test_df9)

test_df9.reset_index(level=0, inplace=True)
test_df9.sort_values(by=["ORGANIZATION_TYPE"],ascending=False,inplace=True)

test_df9.rename(columns=  {"index": "ORGANIZATION_TYPE", 
                     "ORGANIZATION_TYPE":"Default_Percentage"}, 
                                  inplace = True) 


test_df9


# In[109]:


f, ax = plt.subplots(figsize=(22,6))
OrganizationType_vs_Target= sns.barplot(x="ORGANIZATION_TYPE",y="Default_Percentage",data=test_df9
                                         ,order=test_df9['ORGANIZATION_TYPE'])
OrganizationType_vs_Target.set_xticklabels(OrganizationType_vs_Target.get_xticklabels(), rotation=90)
OrganizationType_vs_Target

From above graph, highest number of non-repayment can be seen in Applicants who work in Transport Type3.
# #NAME_HOUSING_TYPE

# In[110]:


#graph to plot housing type of each client in the entire polulation:-

HousingType_vs_Total = sns.countplot("NAME_HOUSING_TYPE",data =app0)
HousingType_vs_Total.set_xticklabels(HousingType_vs_Total.get_xticklabels(),rotation=90)


# In[111]:


#find the percentage housing type of each client the defaulter list:-

test_df10=round((Univariate_defaulter_Cat_1_df["NAME_HOUSING_TYPE"].value_counts()/app0["NAME_HOUSING_TYPE"].value_counts())*100,2)
test_df10 = pd.DataFrame(test_df10)
test_df10.reset_index(level=0, inplace=True)
test_df10.rename(columns=  {"index": "NAME_HOUSING_TYPE", 
                     "NAME_HOUSING_TYPE":"Default_Percentage"}, 
                                 inplace = True) 
test_df10.sort_values(by = 'Default_Percentage' , inplace = True, ascending = False)
test_df10


# In[112]:


#plot to show the housing type of each client client vs. Target variable. Here, Target =1:-

HousingType_vs_Target= sns.barplot(x="NAME_HOUSING_TYPE",y="Default_Percentage",data=test_df10)
HousingType_vs_Target.set_xticklabels(HousingType_vs_Target.get_xticklabels(),rotation=90)

From above graph it can be seen clearly that people with rented apartments are less likely to pay back their loans. This can be because they already have more liabilities compared to other type of people who do not have thia liability.
# # BIVARIATE ANALYSIS

# In[113]:


f, ax = plt.subplots(figsize=(15,10))
sns.scatterplot("AMT_CREDIT","AMT_GOODS_PRICE",data=app0,hue="TARGET")


# # Salary Category vs Clinet who provided Home Number

# In[114]:


f, ax = plt.subplots(figsize=(10,9))
plot_1=sns.barplot("SALARY_CATEGORY","FLAG_WORK_PHONE",data=app0,hue="TARGET")

Client with Extremly low salary has more chance to be a Defaulter, when he did not provide the Home phone number. Here approximately 30% people only produced the phone number
# Salary vs Client whose Permanent Address not match with Contact Address -Region Level

# In[115]:


f, ax = plt.subplots(figsize=(10,9))
plot_1=sns.barplot("SALARY_CATEGORY","REG_REGION_NOT_LIVE_REGION",data=app0,hue="TARGET")

When Client gets Extremply lower salary and if his/her address doest match, then there is a Higher chance for him/her to be defaulter
# # Salary vs Client whose Permanent Address not match with Work Address - Region Level

# In[116]:


f, ax = plt.subplots(figsize=(10,9))
plot_1=sns.barplot("SALARY_CATEGORY","REG_REGION_NOT_WORK_REGION",data=app0,hue="TARGET")

When Client gets Extremply lower salary and if his/her Work address doest match, then there is a Higher chance for him/her to be defaulter
# # Salary vs Client whose Permanent Address not match with Contact Address -City Level

# In[117]:


f, ax = plt.subplots(figsize=(10,9))
plot_1=sns.barplot("SALARY_CATEGORY","REG_CITY_NOT_LIVE_CITY",data=app0,hue="TARGET")

When Client gets LOWER salary and if his/her CONTACT address(CITY-LEVEL)doest match, then there is a Higher chance for him/her to be defaulter
# # Salary vs Client whose Permanent Address not match with Work Address -City Level

# In[118]:


f, ax = plt.subplots(figsize=(10,9))
plot_1=sns.barplot("SALARY_CATEGORY","REG_REGION_NOT_WORK_REGION",data=app0,hue="TARGET")

When Client gets LOWER salary and if his/her WORK address(CITY-LEVEL)doest match, then there is a Higher chance for him/her to be defaulter
# # INCOME TYPE

# Income vs Children count
# 

# In[120]:


f, ax = plt.subplots(figsize=(10,5))
plot_1=sns.barplot("NAME_INCOME_TYPE","CNT_CHILDREN",data=app0,hue="TARGET")
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=90)

People who geting income via Maternity Leave tends to be more Defaulter when they have more children
# # Income vs No.of.FamilyMembers

# In[121]:


f, ax = plt.subplots(figsize=(10,5))
plot_1=sns.barplot("NAME_INCOME_TYPE","CNT_FAM_MEMBERS",data=app0,hue="TARGET")
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=90)

People who geting income via Maternity Leave tends to be more Defaulter when they have more Family Members
# # Income Type vs Client whose Permanent Address not match with Contact Address -Region Level

# In[122]:


f, ax = plt.subplots(figsize=(10,5))
plot_1=sns.barplot("NAME_INCOME_TYPE","REG_REGION_NOT_LIVE_REGION",data=app0,hue="TARGET")
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=90)

Client who are Unemployed has more chance to be a defaulter , when their Permanent Address does not match with the Contact Address in the Regional Level
# # FAMILY STATUS

# Family Status vs Count Of Children

# In[123]:


f, ax = plt.subplots(figsize=(10,5))
plot_1=sns.barplot("NAME_FAMILY_STATUS","CNT_CHILDREN",data=app0,hue="TARGET")
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=90)

Client who are married and has more children (5+), chances to be a defaulter in High. This may be due to the Economic situation of their family, because of more children
# # Family Status vs Count Of Family Members

# In[124]:


f, ax = plt.subplots(figsize=(10,5))
plot_1=sns.barplot("NAME_FAMILY_STATUS","CNT_FAM_MEMBERS",data=app0,hue="TARGET")
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=90)

Client who are married and has more children (5+), chances to be a defaulter in High. This may be due to the Economic situation of their family, because of more children
# In[125]:


f, ax = plt.subplots(figsize=(10,5))
plot_1=sns.barplot("NAME_FAMILY_STATUS","AGE",data=app0,hue="TARGET")
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=90)

Based on the Bivariate analyses, few columns proved to be of no use, so we are dropping them
# In[126]:


app0.drop(["HOUR_APPR_PROCESS_START","FLAG_MOBIL"],axis=1,inplace=True)


# # ANALYSING CORRELATION OF TARGET VARIABLE VS OTHER VARIABLES

# In[127]:


Correlation = app0.corr()
Correlation.sort_values(by=["TARGET"],ascending=False,inplace=True)


# In[128]:


f, ax = plt.subplots(figsize=(20,12))
sns.heatmap(Correlation,annot=True)


# In[129]:


Correlation.head(6)["TARGET"][1:]


# In[130]:


Correlation.tail(5)["TARGET"]


# # Highly Correlated Variables
1.AMT_CREDIT and AMT_GOODS_PRICE =0.99
2.REGION_RATING_CLIENT_W_CITY and REGION_RATING_CLIENT = 0.95
3.CNT_FAM_MEMBERS and CNT_CHILDREN = 0.87
4.AMT_ANNUITY and AMT_CREDIT = 0.77
# # Analysing Previous Application

# In[131]:


#Reading previous.csv Data

pre0.head()


# In[132]:


#find the percentage of null values in each column, to determine what needs to be done as part of clean:-

(pre0.isnull().sum() * 100 / len(pre0)).round(2)


# In[133]:


#method to calculate percentage of NaN values in DataFrame:-

def get_perc_of_missing_values(series):
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 3)
get_perc_of_missing_values(pre0)


# In[134]:


# Iterate over columns in DataFrame and delete those with where >20% of the values are null:-

for col, values in pre0.iteritems():
    if get_perc_of_missing_values(pre0[col]) > 0.20:
        pre0.drop(col, axis=1, inplace=True)
pre0


# In[135]:


#find the percentage of null values in each column, to determine what needs to be done as part of clean:-

(pre0.isnull().sum() * 100 / len(pre0)).round(2)


# In[136]:


# Filling 2% missing value with the Highest Mode in PRODUCT_COMBINATION column:-

pre0["PRODUCT_COMBINATION"].fillna(pre0["PRODUCT_COMBINATION"].mode()[0],inplace=True)


# # Let's start visualising so as to get some viable inference

# Contract Status

# In[137]:


Contract_Status = pre0['NAME_CONTRACT_STATUS']
Contract_Status


# In[138]:


#find the percentage of contract status:-

df_1=round((Contract_Status.value_counts()/pre0["NAME_CONTRACT_STATUS"].count())*100,2)
df_1 = pd.DataFrame(df_1)
df_1.reset_index(level=0, inplace=True)
df_1.rename(columns=  {"index": "NAME_CONTRACT_STATUS", 
                     "NAME_CONTRACT_STATUS":"Percentage_of_Values"}, 
                                 inplace = True) 
df_1.sort_values(by = 'Percentage_of_Values' , inplace = True, ascending = False)
df_1


# In[139]:


# Data to plot:-

labels = 'Approved', 'Canceled', 'Refused', 'Unused offer'
sizes = df_1['Percentage_of_Values']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1, 0.1, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# # Client Type

# In[140]:


Client_Type = pre0['NAME_CLIENT_TYPE']
Client_Type


# In[141]:


#find the percentage of contract status :-

df_2=round((Client_Type.value_counts()/pre0["NAME_CLIENT_TYPE"].count())*100,2)
df_2 = pd.DataFrame(df_2)
df_2.reset_index(level=0, inplace=True)
df_2.rename(columns=  {"index": "NAME_CLIENT_TYPE", 
                     "NAME_CLIENT_TYPE":"Percentage_of_Values"}, 
                                 inplace = True) 
df_2.sort_values(by = 'Percentage_of_Values' , inplace = True, ascending = False)
df_2


# In[142]:


# Data to plot:-

labels = 'Repeater', 'New', 'Refreshed', 'XNA'
sizes = df_2['Percentage_of_Values']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1, 0.1, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# 73.4% applicants are repeaters. Only, 18.4% are new clients.

# In[143]:


Contract_Type = pre0['NAME_CONTRACT_TYPE']
Contract_Type


# In[144]:


#find the percentage of contract status :-

df_3=round((Contract_Type.value_counts()/pre0["NAME_CONTRACT_TYPE"].count())*100,2)
df_3 = pd.DataFrame(df_3)
df_3.reset_index(level=0, inplace=True)
df_3.rename(columns=  {"index": "NAME_CONTRACT_TYPE", 
                     "NAME_CONTRACT_TYPE":"Percentage_of_Values"}, 
                                 inplace = True) 
df_3.sort_values(by = 'Percentage_of_Values' , inplace = True, ascending = False)
df_3


# In[145]:


# Data to :-

labels = 'Cash loans', 'Consumer loans', 'Revolving loans', 'XNA'
sizes = df_3['Percentage_of_Values']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1, 0.1, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# # Days of approval - WEEKDAY_APPR_PROCESS_START

# In[146]:



Approval_days = pre0['WEEKDAY_APPR_PROCESS_START']
Approval_days


# In[147]:


#find the percentage of contract status :-

df_4=round((Approval_days.value_counts()/pre0["WEEKDAY_APPR_PROCESS_START"].count())*100,2)
df_4 = pd.DataFrame(df_4)
df_4.reset_index(level=0, inplace=True)
df_4.rename(columns=  {"index": "WEEKDAY_APPR_PROCESS_START", 
                     "WEEKDAY_APPR_PROCESS_START":"Percentage_of_Values"}, 
                                 inplace = True) 
df_4.sort_values(by = 'Percentage_of_Values' , inplace = True, ascending = False)
df_4


# In[148]:


# Data to plot:-

labels = 'TUESDAY', 'WEDNESDAY', 'MONDAY', 'THURSDAY' , 'FRIDAY' , 'SATURDAY' , 'SUNDAY' 
sizes = df_4['Percentage_of_Values']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue' , '#44FF07' ,'Red','Fuchsia']
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=30)

plt.axis('equal')
plt.show()

Most of the clients have opted to apply loan on Tuesday. It is very interesting to see that applicants are very low on weekends. We would otherwise assume that the applicants would prefer weekends to apply.
# # Purpose of loan - NAME_CASH_LOAN_PURPOSE

# In[149]:


pre0.NAME_CASH_LOAN_PURPOSE


# In[150]:


Loan_Purpose = pre0['NAME_CASH_LOAN_PURPOSE']
Loan_Purpose


# In[151]:


#find the percentage of Loan Purpose:-

df_5=round((Loan_Purpose.value_counts()/pre0["NAME_CASH_LOAN_PURPOSE"].count())*100,2)
df_5 = pd.DataFrame(df_5)
df_5.reset_index(level=0, inplace=True)
df_5.rename(columns=  {"index": "NAME_CASH_LOAN_PURPOSE", 
                     "NAME_CASH_LOAN_PURPOSE":"Percentage_of_Values"}, 
                                 inplace = True) 
df_5.sort_values(by ='Percentage_of_Values', inplace = True, ascending = False)
df_5


# In[152]:


f, ax = plt.subplots(figsize=(10,4))
plot_2=sns.barplot("NAME_CASH_LOAN_PURPOSE","Percentage_of_Values",data=df_5)
plot_2.set_xticklabels(plot_2.get_xticklabels(), rotation=90)

Most Loan purpose was not recorded. XAP and XNA values are highest.
# # Payment type - NAME_PAYMENT_TYPE

# In[154]:


Payment_Type = pre0['NAME_PAYMENT_TYPE']
#find the percentage of Payment Type
df_6=round((Payment_Type.value_counts()/pre0["NAME_PAYMENT_TYPE"].count())*100,2)
df_6 = pd.DataFrame(df_6)
df_6.reset_index(level=0, inplace=True)
df_6.rename(columns=  {"index": "NAME_PAYMENT_TYPE", 
                     "NAME_PAYMENT_TYPE":"Percentage_of_Values"}, 
                                 inplace = True) 
df_6.sort_values(by ='Percentage_of_Values', inplace = True, ascending = False)
df_6


# In[155]:


f, ax = plt.subplots(figsize=(5,4))
plot_2=sns.barplot("NAME_PAYMENT_TYPE","Percentage_of_Values",data=df_6)
plot_2.set_xticklabels(plot_2.get_xticklabels(), rotation=90)

Most people preferred CASH(62.44%) as the mode of Payment
# # Reason of rejection of loan - CODE_REJECT_REASON

# In[156]:


Code_Rejection = pre0['CODE_REJECT_REASON']
#find the percentage of Payment Type
df_7=round((Code_Rejection.value_counts()/pre0["CODE_REJECT_REASON"].count())*100,2)
df_7 = pd.DataFrame(df_7)
df_7.reset_index(level=0, inplace=True)
df_7.rename(columns=  {"index": "CODE_REJECT_REASON", 
                     "CODE_REJECT_REASON":"Percentage_of_Values"}, 
                                 inplace = True) 
df_7.sort_values(by ='Percentage_of_Values', inplace = True, ascending = False)
df_7


# In[157]:


f, ax = plt.subplots(figsize=(5,4))
plot_2=sns.barplot("CODE_REJECT_REASON","Percentage_of_Values",data=df_7)
plot_2.set_xticklabels(plot_2.get_xticklabels(), rotation=90)

Primary reason for the Loan to get rejected is not recorded(XAP (81%)) followed by HC.
# # What kind of goods did the client apply for in the previous application - NAME_GOODS_CATEGORY

# In[158]:


Goods_Category= pre0['NAME_GOODS_CATEGORY']
#find the percentage of Goods Client applied for
df_8=round((Goods_Category.value_counts()/pre0["NAME_GOODS_CATEGORY"].count())*100,2)
df_8 = pd.DataFrame(df_8)
df_8.reset_index(level=0, inplace=True)
df_8.rename(columns=  {"index": "NAME_GOODS_CATEGORY", 
                     "NAME_GOODS_CATEGORY":"Percentage_of_Values"}, 
                                 inplace = True) 
df_8.sort_values(by ='Percentage_of_Values', inplace = True, ascending = False)
df_8


# In[159]:


f, ax = plt.subplots(figsize=(10,5))
plot_2=sns.barplot("NAME_GOODS_CATEGORY","Percentage_of_Values",data=df_8)
plot_2.set_xticklabels(plot_2.get_xticklabels(), rotation=90)


# # Correlation in previous_data df

# In[160]:


Correlation = pre0.corr()
#Correlation.sort_values(by=["TARGET"],ascendingb=False,inplace=True)
f, ax = plt.subplots(figsize=(20,12))
sns.heatmap(Correlation,annot=True)

Above plot shows the Correlation of variables in Previous Application
# In[163]:


prev_current_app_df = pd.merge(pre0,app0,how="inner",on="SK_ID_CURR")
prev_current_app_df.info()


# In[164]:


#find the percentage of null values in each column, to determine what needs to be done as part of clean:-

(prev_current_app_df.isnull().sum() * 100 / len(prev_current_app_df)).round(2)


# # Correlation between previous_data and application_data dataframes

# In[165]:


Correlation = prev_current_app_df.corr()
Correlation.sort_values(by=["TARGET"],ascending=False,inplace=True)
f, ax = plt.subplots(figsize=(20,12))
sns.heatmap(Correlation,annot=True)


# In[166]:


#Above plot is the Correlation between previous_data and application_data dataframes


# In[167]:


Correlation.head(6)["TARGET"][1:]


# In[168]:


Correlation.tail(6)["TARGET"][1:]


# In[ ]:




