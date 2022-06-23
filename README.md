# Stroke-Disease-Classification

## Background
Stroke is a condition that occurs when the blood supply to the brain is interrupted or reduced due to a blockage (ischemic stroke) or rupture of a blood vessel (hemorrhagic stroke). Without blood, the brain will not get oxygen and nutrients, so cells in some areas of the brain will die. This condition causes parts of the body controlled by the damaged area of the brain to not function properly.

Stroke is an emergency condition that needs to be treated as soon as possible, because brain cells can die in just a matter of minutes. Prompt and appropriate treatment measures can minimize the level of brain damage and prevent possible complications.

In this machine learning project, the overall topic that will be resolved is in the health sector regarding stroke, where it will try to predict the possibility of a stroke in a person with certain conditions based on several factors including: age, certain diseases (hypertension, heart disease) who are at high risk of developing stroke. strokes, cigarettes, etc.

As previously explained, stroke can kill the sufferer in a matter of minutes. Detecting stroke with the existing causative factors with the help of machine learning can be very useful in the world of health to detect stroke early in order to increase the sense of heart among sufferers so that strokes can be prevented early.

*References* : [https://www.halodoc.com/kesehatan/stroke](https://www.halodoc.com/kesehatan/stroke)

## Business Understanding
To solve the problem of predicting stroke in a person, machine learning can be done that helps detect the possibility of stroke from the existing signs.
- Problems to solve:
Detection (Prediction) of the possibility of a stroke in a person

- The purpose of making Machine Learning Model:
The model can classify more than 95% of cases with certain conditions

- Solution:
Making Machine Learning with the KNearestNeighbors Algorithm that can classify someone who has the potential to have a stroke

## Data Understanding
The dataset used to predict stroke is a dataset from Kaggle. This dataset has been used to predict stroke with 566 different model algorithms. This dataset has:
- 5110 samples or rows
- 11 features or columns 
- 1 target column (stroke).

This dataset was created by [fedesoriano](https://www.kaggle.com/fedesoriano) and it was last updated 9 months ago.

Dataset: [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

Features in Datasets:
1. id : a unique identifier that distinguishes each data [int]
2. Gender: Patient's gender ('Male', 'Female', and 'Other') [str]
3. age : Age of the patient [int]
4. Hypertension: Hypertension or high blood pressure is a disease that puts a person at risk for stroke. 0 if the patient does not have hypertension, 1 if the patient has hypertension. [int]
5. heart_disease: Heart disease is a disease that puts a person at risk for stroke. 0 if the patient does not have heart disease, 1 if the patient has heart disease. [int]
6. ever_married : Describes whether the patient is married or not ('Yes' or 'No') [str]
7. work_type : Type of employment or status ('children' for children, 'Govt_job' for civil servants, 'Never_worked' for those who have never worked, 'Private' or 'Self-employed' for entrepreneurs or freelancers) [str]
8. Residence_type : Condition of residence ('Rural' for rural areas and 'Urban' for urban areas) [str]
9. avg_glucose_level : Average amount of glucose (sugar) in the blood [float]
10. bmi : Body Mass Index to measure the stability of body weight with height. [float]
11. smoking_status : Description of smoking ('formerly smoked' for those who have smoked, 'never smoked' for those who have never smoked, 'smokes' for those who smoke, and 'unknown' for those whose smoking status is unknown) [str]

Target:
stroke : Prediction target if the patient has a stroke then 1, otherwise 0 [int]

In the project, it can be seen from the number of targets, the number of patients who had a stroke was very small compared to patients who did not have a stroke. And with the heatmap it can be seen that gender does not affect a person can have a stroke or not.

## Data Preparation
For the data preparation stage, several steps have been carried out, namely by overcoming empty data with an average value (*mean substitution*):

`df['bmi'].fillna(df['bmi'].mean(), inplace=True)`

eliminating unnecessary columns such as the 'id' column:

`df = df.drop(['id'], axis=1)`

removing outliers in the data beyond IQR:

`Q1 = df[outlier].quantile(0.25)
Q3 = df[outlier].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[outlier]<(Q1-1.5*IQR))|(df[outlier]>(Q3+1.5*IQR))).any(axis=1)]
df.reset_index(drop=True)`

convert categorical column to numerical:

`df = pd.get_dummies(df)`

using SMOTE Technique to creates as many synthetic examples for minority class as are requirred so that finally two target class are well represented. It does so by synthesising samples that are close to the feature space ,for the minority target class:

`sm = SMOTE(random_state=111)
X_sm , y_sm = sm.fit_resample(X,y)`

dividing the dataset into train and test data:

`X_train, X_test, y_train, y_test = train_test_split(
    X_sm,
    y_sm,
    test_size= .2)`

and standardize X with StandardScaler:

`X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)`

Steps are needed such as filling in empty data with the mean so that the data does not need to be discarded because the mean and median values are almost the same. For the stage of removing outliers and the 'id' column is needed so that data that can damage the model can be removed. After that, it is necessary to divide the dataset into train and test data in order to evaluate the performance of the model with test data that has not been recognized by the model. The standardization stage is carried out so that the features are not slamming in value with other features.

## Modeling
Machine learning modeling to predict stroke in patients uses the K-Nearest Neighbors Classifier algorithm. This algorithm works based on existing features and similarities between these features to classify targets.

`baseline_model = KNeighborsClassifier()`

At the beginning of making the model, the K-NearestNeighborsClassifier model is used with default parameters and when the results are shown the classification achieves an accuracy of 97.5% which has reached the target of model accuracy.

`param_grid = {'n_neighbors': [1, 2],
              'p': [1, 2],
              'weights': ["uniform","distance"],
              'algorithm':["ball_tree", "kd_tree", "brute"],
              }`

`new_param = HalvingGridSearchCV(baseline_model, 
                                param_grid, 
                                cv=StratifiedKFold(n_splits=3, random_state= 123, shuffle=True),
                                resource='leaf_size',
                                max_resources=20,
                                scoring='recall',
                                aggressive_elimination=False).fit(X_train, y_train)`

However, machine learning was developed using HalvingGridSearchCV with several parameters n_neighbors, p, weights, and algorithm. After hyperparameter tuning with these parameters and the scoring category, namely recall, was found the algorithm with the best recall score was found when using algorithm='ball_tree', leaf_size=18, n_neighbors=1, p=1, weights='distance'. Although the model accuracy is drop to 96%, the initial target is a prediction from someone with stroke where the recall value increases from 97.1% to 97.7%. This model can predict the patient with the chance of stroke much better.

`model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=1, p=1, weights='distance')`

## Evaluasi Model
Comparison of metrics between the initial baseline model and the model whose hyperparameters have been tuned with several types of evaluations such as accracy, f1-score, precision, and recall. After hyperparameter tuning of the model, the accuracy of the model decreased by 1% from 97% to 96%, this is because the target to be tuned is the recall score. This is because the model is required to classify patients with possible scores so as to reduce false negatives as much as possible.

Therefore, a model with the best parameters is presented after being tuned with an increase in recall score of 17% compared to the initial baseline model. This model cannot be used to predict the actual data because the precision score of the disease prediction is very low because the number of target datasets is very far apart where the number of non-stroke patients is very large compared to patients who have stroke. Because of this, other algorithms other than KNearestNeighborsClassifier are needed for this stroke data such as LogisticRegression, RandomForest, etc.

### Referensi
- Scikit-learn Docummentation: [https://scikit-learn.org/stable/modules/classes.html](https://scikit-learn.org/stable/modules/classes.html)
- Report Reference: [https://github.com/fahmij8/ML-Exercise/blob/main/MLT-1/MLT_Proyek_Submission_1.ipynb](https://github.com/fahmij8/ML-Exercise/blob/main/MLT-1/MLT_Proyek_Submission_1.ipynb)
- Project: [https://www.kaggle.com/muhamilham/supervised-learning-stroke-prediction](https://www.kaggle.com/muhamilham/supervised-learning-stroke-prediction)
