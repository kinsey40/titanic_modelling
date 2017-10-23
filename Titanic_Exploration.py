#This script is to investigate the Titanic challenge on Kaggle
#Author: Nicholas Kinsey
#Date created: 26/05/2017

#Import relevant libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=0.5)

# Loop over the dataset, looking for titles in name
def Intro_Title_Variable(Dataset_name):

    for i in range(0, len(Dataset_name)):

        if 'Mr.' in Dataset_name.loc[i, 'Name']:
            Dataset_name.loc[i, 'Name_Title'] = 'Mr'
        elif 'Mrs.' in Dataset_name.loc[i, 'Name']:
            Dataset_name.loc[i, 'Name_Title'] = 'Mrs'
        elif 'Miss.' in Dataset_name.loc[i, 'Name']:
            Dataset_name.loc[i, 'Name_Title'] = 'Miss'
        elif 'Master.' in Dataset_name.loc[i, 'Name']:
            Dataset_name.loc[i, 'Name_Title'] = 'Master'
        else:
            Dataset_name.loc[i, 'Name_Title'] = np.nan

    return Dataset_name

def Combine_Two_Variables(Dataset_name, Variable_1, Variable_2):

    if (isinstance(Variable_1, str)) & (isinstance(Variable_2, str)) != True:
        print("Variable Type Error")
        return 1

    New_col = Variable_1 + "_" + Variable_2

    for i in range(0, len(Dataset_name)):

        Dataset_name.loc[i, New_col] = str(Dataset_name.loc[i, Variable_1]) + "_" + str(Dataset_name.loc[i, Variable_2])

    return Dataset_name

def string_char_grab(string, number):
    return string[:number]


#End of functions

#Set the paths to the data
Path_To_Training_Data = os.path.join(os.getcwd(), "Data/Titanic/Titanic_Training.csv")
Path_To_Testing_Data = os.path.join(os.getcwd(), "Data/Titanic/Titanic_Testing.csv")
Path_To_Other_Data = os.path.join(os.getcwd(), "Data/Titanic/Titanic_Gender_Submission.csv")
Path_To_Output = os.path.join(os.getcwd(), "Output/")

# Read the data into a pandas dataset
Training_Dataset = pd.read_csv(Path_To_Training_Data, header=0)
Testing_Dataset = pd.read_csv(Path_To_Testing_Data, header=0)
Answers_Dataset = pd.read_csv(Path_To_Other_Data, header=0)

#Create new column in test data set
Testing_Dataset["Survived"] = np.nan

#Concatanate the datasets, to produce a dataset with all the subjects in
All_Data = pd.concat([Training_Dataset, Testing_Dataset], ignore_index=True)

# Alter the class variable to be a string
All_Data['Pclass'] = All_Data['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})

# Fill missing values in the Survived column and convert to string format
All_Data['Survived'] = All_Data['Survived'].fillna(2.0)
All_Data['Survived'] = All_Data['Survived'].map({1.0: 'Survive', 0.0: 'Died', 2.0: 'Unknown'})

# Print the number of passengers whom survived/died/unknown respectively
#print(All_Data['Pclass'].value_counts())
#print(All_Data['Survived'].value_counts())

# Print the number of passengers whom survived/died/unknown via Pclass
#print(All_Data.groupby(["Pclass", "Survived"]).size())
#print(All_Data.groupby(["Pclass", "Survived"]).size().transform(lambda x: x / sum(x)))

# Alter the class variable to be a string
Training_Dataset['Pclass'] = Training_Dataset['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})
Training_Dataset['Survived'] = Training_Dataset['Survived'].map({1: 'Survive', 0: 'Died'})

# Hypothesis A: Rich people more likely to survive
ax = sns.countplot(x="Pclass", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Pclass_Survival.jpg")
plt.close()

# Hypothesis B: Women more likely to survive
ax = sns.countplot(x="Sex", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Gender_Survival.jpg")
plt.close()

# Hypothesis C: More siblings means better survival
ax = sns.countplot(x="SibSp", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Sibling_Survival.jpg")
plt.close()

# Hypothesis D: Children mean more likely to survive
ax = sns.countplot(x="Parch", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Parent_Survival.jpg")
plt.close()

# Define bins for cont. age variable
bins = [0, 10, 20, 30, 40, 50, 60, Training_Dataset.Age.max()]
group_names = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

# Put age variable into bins
categories = pd.cut(Training_Dataset['Age'], bins, labels=group_names)
Training_Dataset['Age_cats'] = pd.cut(Training_Dataset['Age'], bins, labels=group_names)

# Hypothesis E: Young more likely to survive
ax = sns.countplot(x="Age_cats", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Age_Survival.jpg")
plt.close()

# Check for duplications, identify, examined to be OK
All_Data['Name_Dupe'] = All_Data['Name'].duplicated(False)
Name_Dupes = All_Data.ix[(All_Data['Name_Dupe'] == True)]
All_Data.drop('Name_Dupe', axis=1)

Intro_Title_Variable(All_Data)
Intro_Title_Variable(Training_Dataset)

# Hypothesis F: Titles may indicate no of siblings/spouses
ax = sns.countplot(x="SibSp", hue="Name_Title", data=All_Data)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Siblings_Title.jpg")
plt.close()

# Put age variable into bins, for the All_Data set
categories = pd.cut(All_Data['Age'], bins, labels=group_names)
All_Data['Age_cats'] = pd.cut(All_Data['Age'], bins, labels=group_names)

# Hypothesis G: Age and Titles have some correlation
ax = sns.countplot(x="Age_cats", hue="Name_Title", data=All_Data)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Age_Title.jpg")
plt.close()

# Create a plot, showing how Survivability depends on Pclass and Title
ax = sns.swarmplot(x="Pclass", y="Age", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Age_Pclass_Survived.jpg")
plt.close()

# Create a plot, showing how Survivability depends on Age and Pclass
ax = sns.factorplot(x="Age_cats", col="Pclass", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Age_Pclass_Survived.jpg")
plt.close()

# Create a plot, showing how Survivability depends on Age, Pclass and Title
ax = sns.factorplot(x="Pclass", y="Age", col="Name_Title", hue="Survived", data=Training_Dataset, kind="swarm", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Age_Pclass_Title_Survived.jpg")
plt.close()

# Create a plot, showing how Survivability depends on Pclass and Sex
ax = sns.factorplot(x="Sex", col="Pclass", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Pclass_Sex_Survived.jpg")
plt.close()

# Combine the Pclass and sex into one variable
#Combine_Pclass_Sex_Variables(Training_Dataset)
Combine_Two_Variables(Training_Dataset, "Pclass", "Sex")

# Print out all the missing data for training and combined in Age and title
#print(All_Data['Age'].isnull().sum())
#print(All_Data['Name_Title'].isnull().sum())
#print(Training_Dataset['Age'].isnull().sum())
#print(Training_Dataset['Name_Title'].isnull().sum())

# Create a plot, showing how Survivability depends on Age, Pclass and Sex
ax = sns.factorplot(x="Age_cats", col="Pclass_Sex", col_wrap=3, hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Age_Pclass_Sex_Survived.jpg")
plt.close()

# Look for the data with title of Master as proxy for male children
boys = All_Data.ix[All_Data['Name_Title'] == 'Master']
#print(boys['Age'].describe(), boys['Age'].isnull().sum(), len(boys))

# Title of miss proxy for female children
misses = All_Data.ix[All_Data['Name_Title'] == 'Miss']
#print(misses['Age'].describe(), misses['Age'].isnull().sum(), len(misses))

# Plot the distribution across age and pclass of the 'miss' passengers
ax = sns.factorplot(x="Age_cats", col="Pclass", col_wrap=3, hue="Survived", data=misses, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Misses_Age_Pclass_Survived.jpg")
plt.close()

# Look for the women travelling alone; no parents, siblings, children
misses_alone = misses.ix[(misses['SibSp'] == 0) & (misses['Parch'] == 0)]
#print(misses_alone['Age'].describe(), misses_alone['Age'].isnull().sum(), len(misses_alone))

# Look for the number of records of women travelling alone whom are less than 14.5 yrs old
#print(len(misses_alone.ix[misses_alone['Age'] <= 14.5]))

# Therefore, safe to assume, a female travelling alone is NOT a child

# Look at the SibSp variable
#print(All_Data['SibSp'].describe(), All_Data['SibSp'].isnull().sum(), len(All_Data))

# Make the SibSp variable categorical
All_Data['SibSp'] = All_Data['SibSp'].astype('category')
Training_Dataset['SibSp'] = Training_Dataset['SibSp'].astype('category')

# Call the combine two variables function
Combine_Two_Variables(Training_Dataset, "Pclass", "Name_Title")

# Create a plot of Sibsp survivability dependant on Pclass and Title
ax = sns.factorplot(x="SibSp", col="Pclass_Name_Title", col_wrap=6, hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Pclass_Title_Sibsp_Survived.jpg")
plt.close()

# Categorise the Parch variable
All_Data['Parch'] = All_Data['Parch'].astype('category')
Training_Dataset['Parch'] = Training_Dataset['Parch'].astype('category')

# Create a plot of Parch survivability dependant on Pclass and Title
ax = sns.factorplot(x="Parch", col="Pclass_Name_Title", col_wrap=4, hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Pclass_Title_Parch_Survived.jpg")
plt.close()

#Convert SibSp and Parch to numeric
All_Data['SibSp'] = All_Data['SibSp'].astype('int64')
All_Data['Parch'] = All_Data['Parch'].astype('int64')
Training_Dataset['SibSp'] = Training_Dataset['SibSp'].astype('int64')
Training_Dataset['Parch'] = Training_Dataset['Parch'].astype('int64')

# Create new variable defining the size of a family
All_Data['Family_Size'] = All_Data['SibSp'] + All_Data['Parch'] + 1
All_Data['Family_Size'] = All_Data['Family_Size'].astype('category')
Training_Dataset['Family_Size'] = Training_Dataset['SibSp'] + Training_Dataset['Parch'] + 1
Training_Dataset['Family_Size'] = Training_Dataset['Family_Size'].astype('category')

# Create plot showing relation between family size, pclass, title and survivability
ax = sns.factorplot(x="Family_Size", col="Pclass_Name_Title", col_wrap=4, hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Pclass_Title_FamSize_Survived.jpg")
plt.close()

All_Data['Ticket'] = All_Data['Ticket'].astype('str')
All_Data['Ticket'].fillna(" ", inplace=True)
All_Data['Ticket_First_Char'] = All_Data['Ticket'].map(lambda x: string_char_grab(x,1))

Training_Dataset['Ticket'] = Training_Dataset['Ticket'].astype('str')
Training_Dataset['Ticket'].fillna(" ", inplace=True)
Training_Dataset['Ticket_First_Char'] = Training_Dataset['Ticket'].map(lambda x: string_char_grab(x,1))

Training_Dataset['Ticket_First_Char'] = Training_Dataset['Ticket_First_Char'].astype('category')

# Create a plot of Ticket FC survivability
ax = sns.countplot(x="Ticket_First_Char", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Ticket_FC.jpg")
#plt.show()
plt.close()


# Create a plot of Ticket FC survivability dependant on Pclass
ax = sns.factorplot(x="Ticket_First_Char", col="Pclass", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Ticket_FC_Pclass.jpg")
plt.close()
#plt.show()

# Create a plot of Ticket FC survivability dependant on Pclass and title
ax = sns.factorplot(x="Ticket_First_Char", col="Pclass_Name_Title", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Ticket_FC_Pclass_Title.jpg")
plt.close()
#plt.show()

sns.distplot(All_Data['Fare'].dropna(), kde=False)
plt.close()

# Create a plot of fare survivability
ax = sns.factorplot(x="Fare", col="Pclass_Name_Title", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Fare_Pclass_Title_Survived.jpg")
plt.close()
#plt.show()

All_Data['Cabin'].fillna(value='U', inplace=True)
All_Data['Cabin'] = All_Data['Cabin'].astype('str')

All_Data['Cabin_First_Char'] = All_Data['Cabin'].map(lambda x: string_char_grab(x,1))
All_Data['Cabin_First_Char'] = All_Data['Cabin_First_Char'].astype('category')

Training_Dataset['Cabin'].fillna(value='U', inplace=True)
Training_Dataset['Cabin'] = Training_Dataset['Cabin'].astype('str')

Training_Dataset['Cabin_First_Char'] = Training_Dataset['Cabin'].map(lambda x: string_char_grab(x,1))
Training_Dataset['Cabin_First_Char'] = Training_Dataset['Cabin_First_Char'].astype('category')

# Create plot of cabin survivability
ax = sns.countplot(x="Cabin_First_Char", hue="Survived", data=Training_Dataset)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Cabin_Survived.jpg")
#plt.show()
plt.close()

# Create plot of cabin and Pclass survivability
ax = sns.factorplot(x="Cabin_First_Char", col="Pclass", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Cabin_Pclass_Survived.jpg")
#plt.show()
plt.close()

# Introduce a multiple cabins variable
All_Data['Multiple_Cabins'] = np.where(All_Data['Cabin'].str.contains(" ") == True, 'Y', 'N')
Training_Dataset['Multiple_Cabins'] = np.where(Training_Dataset['Cabin'].str.contains(" ") == True, 'Y', 'N')

# Create plot of Multiple cabins and Pclass, Title survivability
ax = sns.factorplot(x="Multiple_Cabins", col="Pclass_Name_Title", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Multiple_Cabin_Pclass_Title_Survived.jpg")
#plt.show()
plt.close()

# Look at the embarked variable
Training_Dataset['Embarked'] = Training_Dataset['Embarked'].astype('category')
All_Data['Embarked'] = All_Data['Embarked'].astype('category')

# Create plot of Embarked and Pclass, Title survivability
ax = sns.factorplot(x="Embarked", col="Pclass_Name_Title", hue="Survived", data=Training_Dataset, kind="count", legend=False)
plt.legend(loc='upper right')
plt.savefig(Path_To_Output + "Embarked_Pclass_Title_Survived.jpg")
#plt.show()
plt.close()
