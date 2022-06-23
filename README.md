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
For the data preparation stage, several steps have been carried out, namely by overcoming empty data with an average value (*mean substitution*), eliminating unnecessary columns such as the 'id' column, removing outliers in the data with IQR, dividing the dataset into train and test data, and standardize with StandardScaler.

Steps are needed such as filling in empty data with the mean so that the data does not need to be discarded because the mean and median values are almost the same. For the stage of removing outliers and the 'id' column is needed so that data that can damage the model can be removed. After that, it is necessary to divide the dataset into train and test data in order to evaluate the performance of the model with test data that has not been recognized by the model. The standardization stage is carried out so that the features are not slamming in value with other features.

## Modeling
Seperti yang sudah dijelaskan diawal, pemodelan machine learning untuk memprediksi stroke pada pasien yaitu menggunakan algoritma K-Nearest Neighbors Classifier. Algoritma ini bekerja berdasarkan dengan fitur fitur yang ada dan kemiripan antara fitur fitur tersebut untuk mengklasifikasikan target.

Dengan data yang ada dan setelah dilakukan pengolahan data, diambil fitur fitur yang berpengaruh tinggi terhadap kemungkinan seseorang terkena stroke. Beberapa fitur yang digunakan yaitu seperti usia, hipertensi, penyakit jantung, pekerjaan, dan status merokok.

Pada awal pembuatan model, digunakan model K-NearestNeighborsClassifier dengan parameter default dan saat ditunjukkan hasil dari klasifikasinya meraih akurasi sebesar 96% dimana telah mencapai target dari akurasi model.

Tetapi dilakukan pengembangan machine learning menggunakan HalvingGridSearchCV dengan beberapa parameter n_neighbors, p, weights, dan algorithm. Setelah dilakukan hyperparameter tuning dengan parameter tersebut dan dengan scoring category yaitu recall, ditemukan algoritma dengan score recall terbaik yaitu saat menggunakan algorithm='brute', leaf_size=18, n_neighbors=1, p=1, weights='distance'. Meskipun didapatkan akurasi model sebesar 94%, namun target awal merupakan prediksi dari seseorang dengan penyakit stroke dimana nilai recall naik sebesar 0.17.

## Evaluasi Model
Dilakukan perbandingan metrics antara model baseline awal dengan model yang hyperparameternya telah dituning dengan beberapa jenis evaluasi seperti accracy, f1-score, precision, dan recall. Setelah model dilakukan hyperparameter tuning, akurasi dari model berkurang sebesar 2% dari 96% menjadi 94%, hal ini dikarenakan target yang ingin dituning merupakan recall score.

Hal ini karena model diharusikan untuk mengklasifikasikan pasien dengan kemungkinan score sehingga sebisa mungkin mengurangi false negative.

Maka dari itu disajikan model dengan parameter terbaik setelah dituning dengan kenaikan recall score sebesar 17% dibandingkan dengan model baseline awal. Model ini belum dapat digunakan untuk memprediksi data sesungguhnya karena precision score dari prediksi penyakit sangat rendah karena jumlah target dataset yang sangat yang berjauhan dimana jumlah pasien non stroke sangat banyak dibandingkan dengan pasien yang terkena stroke. Karena hal ini dibutuhkan algoritma lain selain dari KNearestNeighborsClassifier untuk data stroke ini seperti LogisticRegression, RandomForest, dll.

### Referensi
- Dokumentasi Scikit-learn: [https://scikit-learn.org/stable/modules/classes.html](https://scikit-learn.org/stable/modules/classes.html)
- Referensi Laporan: [https://github.com/fahmij8/ML-Exercise/blob/main/MLT-1/MLT_Proyek_Submission_1.ipynb](https://github.com/fahmij8/ML-Exercise/blob/main/MLT-1/MLT_Proyek_Submission_1.ipynb)
- Projek: [https://www.kaggle.com/muhamilham/supervised-learning-stroke-prediction](https://www.kaggle.com/muhamilham/supervised-learning-stroke-prediction)
