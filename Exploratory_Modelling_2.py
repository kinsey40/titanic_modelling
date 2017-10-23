# The second file for exploratory modelling as the first one got over-written

#Import relevant libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import RepeatedStratifiedKFold

pd.options.mode.chained_assignment = None  # default='warn'

def read_in_data():

    Path_To_Training_Data = os.path.join(os.getcwd(), "Data/Titanic/Titanic_Training.csv")
    Path_To_Testing_Data = os.path.join(os.getcwd(), "Data/Titanic/Titanic_Testing.csv")
    Path_To_Other_Data = os.path.join(os.getcwd(), "Data/Titanic/Titanic_Gender_Submission.csv")

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

    # Alter the class variable to be a string
    Training_Dataset['Pclass'] = Training_Dataset['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})
    #Training_Dataset['Survived'] = Training_Dataset['Survived'].map({1: 'Survive', 0: 'Died'})

    return [All_Data, Training_Dataset, Testing_Dataset]

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

def create_new_variables(data):

    # Define bins for cont. age variable
    bins = [0, 10, 20, 30, 40, 50, 60, data.Age.max()]
    group_names = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

    # Put age variable into bins
    categories = pd.cut(data['Age'], bins, labels=group_names)
    data['Age_cats'] = pd.cut(data['Age'], bins, labels=group_names)

    for i in range(0, len(data)):

        if 'Mr.' in data.loc[i, 'Name']:
            data.loc[i, 'Name_Title'] = 'Mr'
        elif 'Mrs.' in data.loc[i, 'Name']:
            data.loc[i, 'Name_Title'] = 'Mrs'
        elif 'Miss.' in data.loc[i, 'Name']:
            data.loc[i, 'Name_Title'] = 'Miss'
        elif 'Master.' in data.loc[i, 'Name']:
            data.loc[i, 'Name_Title'] = 'Master'
        else:
            data.loc[i, 'Name_Title'] = np.nan

    # Combine Pclass and sex
    Combine_Two_Variables(data, "Pclass", "Sex")

    # Call the combine two variables function
    Combine_Two_Variables(data, "Pclass", "Name_Title")

    #Convert SibSp and Parch to numeric
    data['SibSp'] = data['SibSp'].astype('int64')
    data['Parch'] = data['Parch'].astype('int64')

    # Create new variable defining the size of a family
    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
    data['Family_Size'] = data['Family_Size'].astype('category')

    return data

def create_model(x_train, y_train, model_save_loc, hit, no_hit):

    cw = {0: hit, 1: no_hit}

    clf = RandomForestClassifier(n_estimators=1000, criterion='gini', oob_score=True, max_features='auto', class_weight=cw)

    model = clf.fit(x_train, y_train)

    # Save the model
    joblib.dump(model, model_save_loc)

def evaluate_model(x_test, y_test, imp_plot_save_loc, model_save_loc):

    model = joblib.load(model_save_loc)

    oob_score = model.oob_score_
    feature_imp = model.feature_importances_

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(x_test.shape[1]), feature_imp,
           color="r", align="center")
    plt.yticks(range(x_test.shape[1]), x_test)
    #plt.show()
    plt.savefig(imp_plot_save_loc)

    preds = model.predict(x_test)
    conf_mat = pd.crosstab(y_test, preds, rownames=['actual'], colnames=['preds'], normalize=True)

    return conf_mat, oob_score, feature_imp

def apply_cv(df, target_var, feature_vars, n_folds, imp_plot_save_loc, model_save_loc):

    feature_vars.append(target_var)
    df_select = df[feature_vars]
    feature_vars.remove(target_var)

    y, _ = pd.factorize(df_select[target_var])
    features = pd.DataFrame()

    for variable in feature_vars:
        new_var = variable + "_cat"
        df_select[variable] = df_select[variable].astype('category')
        df_select[new_var] = df_select[variable].cat.codes
        features = pd.concat([features, df_select[new_var]], axis=1)

    no_hit = len(df_select.ix[df_select[target_var] == 0]) / len(df_select)
    hit = len(df_select.ix[df_select[target_var] == 1]) / len(df_select)

    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=10)
    oob_scores = []

    features_np = features.as_matrix()

    for train_index, test_index in rskf.split(features_np, y):
        x_train, x_test = features_np[train_index], features_np[test_index]
        y_train, y_test = y[train_index], y[test_index]
        create_model(pd.DataFrame(x_train, columns=feature_vars), pd.DataFrame(y_train, columns=[target_var]), model_save_loc, hit, no_hit)
        cnf_mat, oob_score, f_imp = evaluate_model(pd.DataFrame(x_test, columns=feature_vars), pd.DataFrame(y_test, columns=[target_var]), imp_plot_save_loc, model_save_loc)
        oob_scores.append(oob_score)

    oob_scores_np = np.asarray(oob_scores)
    mean_oob = np.mean(oob_scores_np)
    print(oob_scores)

    return mean_oob

if __name__ == '__main__':

    #Set the paths to the data
    Path_To_Output = os.path.join(os.getcwd(), "Output/")
    model_save_loc = os.path.join(os.getcwd(), "Output", "model.pk1")
    imp_plot_save_loc = os.path.join(os.getcwd(), "Output", "feature_vars_imp_plot.jpg")

    data_list = read_in_data()
    n_folds = 3

    all_data = create_new_variables(data_list[0])
    train_data = create_new_variables(data_list[1])
    test_data = create_new_variables(data_list[2])

    target_var = "Survived"
    feature_vars = ["Pclass", "Name_Title", "Family_Size"]

    #create_model(train_data, feature_vars, target_var, model_save_loc)
    #conf_mat, oob_score, feature_imp = evaluate_model(train_data, target_var, feature_vars, imp_plot_save_loc, model_save_loc)

    #print(conf_mat, oob_score, feature_imp)
    output = apply_cv(train_data, target_var, feature_vars, n_folds, imp_plot_save_loc, model_save_loc)
    print(output)
