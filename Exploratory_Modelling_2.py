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
from sklearn import tree
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
import _pickle as cPickle
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import subprocess

pd.options.mode.chained_assignment = None  # default='warn'

def read_in_data():

    Path_To_Training_Data = os.path.join(os.getcwd(), "titanic_modelling/Data/Titanic/Titanic_Training.csv")
    Path_To_Testing_Data = os.path.join(os.getcwd(), "titanic_modelling/Data/Titanic/Titanic_Testing.csv")
    Path_To_Other_Data = os.path.join(os.getcwd(), "titanic_modelling/Data/Titanic/Titanic_Gender_Submission.csv")

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
    All_Data['Survived'] = All_Data['Survived'].map({1.0: 'Survive', 0.0: 'Died', 2.0: 'Unknown'})
    All_Data['Survived'] = All_Data['Survived'].fillna(2.0)

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

    mr_title_list = ['Mr.', 'Rev.', 'Capt.', 'Col.', 'Sir.', 'Major.', 'Don.', 'Jonkheer.', 'Dr.']
    mrs_title_list = ['Mrs.', 'Lady.', 'Countess.', 'Dona.']
    miss_title_list = ['Miss.', 'Ms.', 'Mlle.']

    for i in range(0, len(data)):

        if any(x in data.loc[i, 'Name'] for x in mr_title_list):
            data.loc[i, 'Name_Title'] = 'Mr'

        elif any(x in data.loc[i, 'Name'] for x in mrs_title_list):
            data.loc[i, 'Name_Title'] = 'Mrs'

        elif any(x in data.loc[i, 'Name'] for x in miss_title_list):
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

    # Creates a new variable which counts the freq of each ticket
    data['Ticket_P_Size'] = data.groupby('Ticket')['Ticket'].transform('count')

    # Create an average fare for all the people on that ticket
    data["Ave_Fare"] = data["Fare"] / data["Ticket_P_Size"]

    return data

def create_model(x_train, y_train, model_save_loc):

    cw_np = compute_class_weight('balanced', np.unique(y_train), y_train.as_matrix().flatten())
    cw_dict = {0: cw_np[0], 1: cw_np[1]}

    clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto', class_weight=cw_dict)

    y_train_alt = np.ravel(y_train)
    clf = clf.fit(x_train, y_train_alt)

    # Save the model
    with open(model_save_loc, 'wb') as f:
        cPickle.dump(clf, f)

def evaluate_model(x_test, y_test, imp_plot_save_loc, model_save_loc, tree_save_location, feature_vars):

    with open(model_save_loc, 'rb') as f:
        model = cPickle.load(f)

    i_tree = 0
    for tree_in_forest in model.estimators_:
        with open(tree_save_location, 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
        i_tree = i_tree + 1

    tree_file_name = tree_save_location.split('.')[0]
    new_tree_file_name = tree_file_name + '.png'
    subprocess.call(['dot', '-Tpng', tree_save_location, '-o', new_tree_file_name])

    feature_imp = model.feature_importances_

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(x_test.shape[1]), feature_imp,
           color="r", align="center")
    plt.yticks(range(x_test.shape[1]), x_test)
    plt.xlabel("Relative Importance")
    plt.savefig(imp_plot_save_loc)
    plt.close()

    preds = model.predict(x_test)
    y_test_mat = y_test.as_matrix().reshape(preds.shape)

    #print(preds.shape, y_test)
    conf_mat = pd.crosstab(y_test_mat, preds, rownames=['actual'], colnames=['preds'], normalize=True)

    return conf_mat, feature_imp

def apply_cv(df, target_var, feature_vars, n_folds, imp_plot_save_loc, model_save_loc, tree_plot_save_loc):

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

    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=10)

    features_np = features.as_matrix()

    for train_index, test_index in rskf.split(features_np, y):
        x_train, x_test = features_np[train_index], features_np[test_index]
        y_train, y_test = y[train_index], y[test_index]
        create_model(pd.DataFrame(x_train, columns=feature_vars), pd.DataFrame(y_train, columns=[target_var]), model_save_loc)
        cnf_mat, f_imp = evaluate_model(pd.DataFrame(x_test, columns=feature_vars), pd.DataFrame(y_test, columns=[target_var]), imp_plot_save_loc, model_save_loc, tree_plot_save_loc, feature_vars)

    acc = cnf_mat.iloc[0,0] + cnf_mat.iloc[1,1]

    return cnf_mat, f_imp, acc

def plot_fare_density(data, fares_dens_plot_save_loc):

    all_data = data.ix[(data["Name_Title"] == "Mr") & (data["Pclass"] == 'First')]

    all_data_surv = all_data.ix[all_data["Survived"] == "Survive"]
    all_data_died = all_data.ix[all_data["Survived"] == "Died"]
    all_data_unknown = all_data.ix[all_data["Survived"] == 2]

    plt.figure()
    sns.kdeplot(all_data_surv["Fare"].values, shade=True, label="Survived")
    sns.kdeplot(all_data_died["Fare"].values, shade=True, label="Died")
    sns.kdeplot(all_data_unknown["Fare"].values, shade=True, label="Unknown")
    plt.xlabel("Fare (£)")
    plt.ylabel("Norm. Density")
    plt.title("Dist. of fare, men in 1st class")
    plt.savefig(fares_dens_plot_save_loc)
    plt.close()

def plot_ticket_p_size(data, save_loc):

    all_data = data.ix[(data["Name_Title"] == "Mr") & (data["Pclass"] == 'First')]
    all_data["Ticket_P_Size"].astype('category')

    all_data_surv = all_data.ix[all_data["Survived"] == "Survive"]
    all_data_died = all_data.ix[all_data["Survived"] == "Died"]
    all_data_unknown = all_data.ix[all_data["Survived"] == 2]

    plt.figure()
    sns.countplot(x='Ticket_P_Size', hue="Survived", data=all_data)
    plt.xlabel("Party Size")
    plt.ylabel("Count")
    plt.title("Ticket Party Size of men in First Class")
    plt.savefig(save_loc)
    plt.close()

def missing_fare(data):

    null_data = data[data["Ave_Fare"].isnull()]
    missing_data_values = []
    new_fares = []
    loc_missing_values = null_data.index.values.tolist()

    for index, value in null_data.iterrows():
        ticket_value = value['Ticket']
        pclass_value = value['Pclass_Name_Title']
        family_value = value['Family_Size']
        missing_data_values.append([ticket_value, pclass_value, family_value])

    for ticket_value, pclass_value, family_value in missing_data_values:
        data_comparison = data.ix[(data['Pclass_Name_Title'] == pclass_value) & (data['Family_Size'] == family_value) & (data['Ticket'] != ticket_value)]
        calc_fare = data_comparison["Ave_Fare"].median()
        new_fares.append(calc_fare)

    for index, value in zip(loc_missing_values, new_fares):
        data.at[index, "Ave_Fare"] = value

    return data

def processing_data(data, cols):

    new_data = preprocessing.scale(data[cols])

    for i, col in enumerate(cols):
        data[col] = new_data[:,i]

    calc_corr = data[cols[0]].corr(data[cols[1]])

    return data, calc_corr

if __name__ == '__main__':

    #Set the paths to the data
    Path_To_Output = os.path.join(os.getcwd(), "titanic_modelling/Output/")
    model_save_loc = os.path.join(os.getcwd(), "titanic_modelling/Output", "model.pk1")
    imp_plot_save_loc = os.path.join(os.getcwd(), "titanic_modelling/Output", "feature_vars_imp_plot.jpg")
    tree_plot_save_loc = os.path.join(os.getcwd(), "titanic_modelling/Output/", "tree_visualisation.dot")
    fares_dens_plot_save_loc = os.path.join(os.getcwd(), "titanic_modelling/Output/", "fare_density_plot.jpg")
    bar_p_size_loc = os.path.join(os.getcwd(), "titanic_modelling/Output/", "ticket_p_size_survivability.jpg")


    data_list = read_in_data()
    n_folds = 3

    all_data = create_new_variables(data_list[0])
    train_data = create_new_variables(data_list[1])
    test_data = create_new_variables(data_list[2])

    target_var = "Survived"
    feature_vars = ["Name_Title", "Ticket_P_Size", "Ave_Fare"]

    all_data = missing_fare(all_data)
    train_data = missing_fare(train_data)
    test_data = missing_fare(test_data)

    returned_data, corr = processing_data(all_data, ["Ave_Fare", "Ticket_P_Size"])
    returned_data, corr = processing_data(train_data, ["Ave_Fare", "Ticket_P_Size"])
    returned_data, corr = processing_data(test_data, ["Ave_Fare", "Ticket_P_Size"])


    confusion_matrix, feature_importances, accuracy = apply_cv(train_data, target_var, feature_vars, n_folds, imp_plot_save_loc, model_save_loc, tree_plot_save_loc)
    print(confusion_matrix, accuracy, feature_importances)
    #plot_fare_density(all_data, fares_dens_plot_save_loc)
    #plot_ticket_p_size(all_data, bar_p_size_loc)
