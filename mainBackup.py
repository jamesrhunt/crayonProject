########################################################################

# ENCODING CATEGORICAL DATA 

# Encode Attrition to 1, 0 (1 = Attrition)
y_df = (y_df.astype('category')).cat.codes
print(y_df)


# one hot encode all categorical features:
x_dfCategorical = x_df[categoricalFeatureNames]
print(x_dfCategorical.columns)
#x_dfCategorical = pd.get_dummies(x_dfCategorical.replace(' ', '_', regex=True), drop_first=True)
# use pandas' basic get dummies function, can switch between drop_first
x_dfCategorical = pd.get_dummies(x_dfCategorical, drop_first=True)
categoricalFeatureNames= x_dfCategorical.columns
print(categoricalFeatureNames)


# recombine with numerical data for split and scale
# Select numerical data
x_dfNumerical = x_df[numericalFeatureNames]
x_df = pd.concat([x_dfNumerical, x_dfCategorical], axis=1)
# save the feature names again


#for col, val in x_df.items():
#    print(col,val)

########################################################################

# SCALING NUMERICAL DATA


# TRAIN TEST SPLIT BEFORE SCALING 

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42, stratify=y_df)
print(x_train, y_train)

print(type(x_train[numericalFeatureNames]))
print((x_train[numericalFeatureNames].shape))

# scale numeric data in training and test set (no fit for test set)
scaler = StandardScaler()
x_scaled_train = scaler.fit_transform(x_train[numericalFeatureNames])
x_scaled_test = scaler.transform(x_test[numericalFeatureNames])
#print(x_scaled_train)
#print(x_scaled_test)
print(type(x_scaled_train))
print((x_scaled_train.shape))


# reconvert to dataframe containing numeric scaled data
x_dfScaledNum_train = pd.DataFrame(x_scaled_train, columns=numericalFeatureNames)
x_dfScaledNum_test = pd.DataFrame(x_scaled_test, columns=numericalFeatureNames)

# reset indicies of dataframes before recombining
x_dfScaledNum_train.reset_index(drop=True, inplace=True)
x_dfScaledNum_test.reset_index(drop=True, inplace=True)
x_dfCat_train = x_train[categoricalFeatureNames]
x_dfCat_test = x_test[categoricalFeatureNames]
x_dfCat_train.reset_index(drop=True, inplace=True)
x_dfCat_test.reset_index(drop=True, inplace=True)

# recombine into dataframes containing the scaled numerical featuers and the encoded categorical featuers
x_dfScaled_train = pd.concat([x_dfScaledNum_train, x_dfCat_train], axis=1)
x_dfScaled_test = pd.concat([x_dfScaledNum_test, x_dfCat_test], axis=1)


print(x_dfScaled_train)
print(x_dfScaled_test)
print(type(x_dfScaled_train))
print((x_dfScaled_train.shape))
print(type(x_dfScaled_test))
print((x_dfScaled_test.shape))

if 0:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(x_dfScaled_train, title="Scaled Data")
    profile.to_file("data/reports/scaledData.html")



########################################################################

# TRAINING AND TESTING MODELS
from sklearn.neighbors import KNeighborsClassifier

useSMOTE = True
if useSMOTE:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    x_dfScaled_train, y_train = smote.fit_resample(x_dfScaled_train, y_train)

########################################################################

# FEATURE SELECTION

from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, RFE
from sklearn.ensemble import RandomForestClassifier

if 0:
    # SelectKBest
    selector = SelectKBest(f_classif, k=40)  # Select the top k features
    x_selected = selector.fit_transform(x_dfScaled_train, y_train)

    # Get the selected feature indices
    selected_indices = selector.get_support()

    # Get the selected feature names from the original DataFrame
    selected_feature_names = x_dfScaled_train.columns[selected_indices]
    print(selected_feature_names)

    # select the "selected" columns from the dataframe
    x_dfScaled_train = x_dfScaled_train[selected_feature_names]
    x_dfScaled_test = x_dfScaled_test[selected_feature_names]


# Set up a KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the classifier to the training data
knn.fit(x_dfScaled_train,y_train)
y_pred = knn.predict(x_dfScaled_test)

# Metrics

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

quit()
#neighbors = [2,4,6,8,10,12,14,16,18,20]
neighbors = [5]

train_accuracies = {}
test_accuracies = {}
for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)
  
    # Fit the classifier to the training data
    knn.fit(x_dfScaled_train,y_train)
    y_pred = knn.predict(x_dfScaled_test)
    # Compute accuracy
    train_accuracies[neighbor] = knn.score(x_dfScaled_train, y_train)
    test_accuracies[neighbor] = knn.score(x_dfScaled_test, y_test)
    
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)

if 0:
    plt.title("KNN: Varying Number of Neighbors")
    # Plot training accuracies
    plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
    # Plot test accuracies
    plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
    plt.legend()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    # Display the plot
    plt.show()

# Print the accuracy
print(knn.score(x_dfScaled_test, y_test))