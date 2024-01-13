import re
import csv
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Specify file paths directly
training_path = 'C:/Users/othma/OneDrive/Documents/Python_Workspace/Project/venv/Training.csv'
testing_path = 'C:/Users/othma/OneDrive/Documents/Python_Workspace/Project/venv/Testing.csv'

# Read the CSV files
training = pd.read_csv(training_path)
testing = pd.read_csv(testing_path)

cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print("Decision Tree Classifier Cross Validation Scores:", scores.mean())

model = SVC(probability=True)
model.fit(x_train, y_train)
svm_score = model.score(x_test, y_test)
print("SVM Classifier Score:", svm_score)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        print("You should take consultation from a doctor.")
    else:
        print("It might not be that bad, but you should take precautions.")

def getDescription():
    global description_list
    file_path = 'C:/Users/othma/OneDrive/Documents/Python_Workspace/Project/venv/symptom_Description.csv'
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    file_path = 'C:/Users/othma/OneDrive/Documents/Python_Workspace/Project/venv/Symptom_severity.csv'
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    file_path = 'C:/Users/othma/OneDrive/Documents/Python_Workspace/Project/venv/symptom_precaution.csv'
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    file_path = 'C:/Users/othma/OneDrive/Documents/Python_Workspace/Project/venv/Training.csv'
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("Searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter a valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. For how many days have you been experiencing this symptom? : "))
            break
        except:
            print("Enter a valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any of the following symptoms?")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? (yes/no) : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("Provide a proper answer (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            print("Take the following measures : ")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)

    recurse(0, 1)

getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
print("----------------------------------------------------------------------------------------")


import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import numpy as np  

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

#  Feature Importance Plot
def plot_feature_importances(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(np.arange(len(indices)), sorted_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
    plt.tight_layout()
    plt.show()

#  Confusion Matrix Plot
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
# Plot Feature Importances for Decision Tree
def plot_feature_importances(model, features):
    n_features = len(features)
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(clf, features)

# Plot Confusion Matrix for Decision Tree Classifier
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix for Decision Tree
dt_cm = confusion_matrix(y_test, clf.predict(x_test))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix for Decision Tree
plt.figure()
plot_confusion_matrix(dt_cm, classes=le.classes_,
                      title='Confusion matrix for Decision Tree, without normalization')

# Plot normalized confusion matrix for Decision Tree
plt.figure()
plot_confusion_matrix(dt_cm, classes=le.classes_, normalize=True,
                      title='Normalized confusion matrix for Decision Tree')

# Compute confusion matrix for SVM
svm_cm = confusion_matrix(y_test, model.predict(x_test))

# Plot non-normalized confusion matrix for SVM
plt.figure()
plot_confusion_matrix(svm_cm, classes=le.classes_,
                      title='Confusion matrix for SVM, without normalization')

# Plot normalized confusion matrix for SVM
plt.figure()
plot_confusion_matrix(svm_cm, classes=le.classes_, normalize=True,
                      title='Normalized confusion matrix for SVM')

# Enhanced ROC Curve Plot
def plot_roc_curves(y_test, y_score, classes):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

# Calculate probability estimates for SVM on the test data
svm_probs = model.predict_proba(x_test)

# ROC Curve for SVM (One-vs-Rest approach)
y_test_bin = label_binarize(y_test, classes=le.classes_)
n_classes = y_test_bin.shape[1]

# Initialize a dictionary to store ROC AUC scores for each class
roc_auc_dict = {}

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], svm_probs[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[le.classes_[i]] = roc_auc
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc, le.classes_[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for SVM (One-vs-Rest)')
plt.legend(loc="lower right")

# Show ROC AUC scores for each class
for class_name, roc_auc in roc_auc_dict.items():
    print(f"ROC AUC for class {class_name}: {roc_auc:.2f}")

# Show all plots
plt.show()
