import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_wine
data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df.info())

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=i)
    
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],  
    }
    
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    best_par = grid_search.best_par_
    best_accuracy = grid_search.best_score_
    
    results.append({
        'Sample': f'S{i+1}',
        'Best Accuracy': best_accuracy,
        'Best Parameters': best_par
    })

results_df = pd.DataFrame(results)
print(results_df)

best_sample_index = results_df['Best Accuracy'].idxmax()
best_par = results[best_sample_index]['Best Parameters']

best_svm = SVC(**best_par)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=best_sample_index)

iterations = 1000
accuracies = []

min_samples_per_class = 10
for iteration in range(min_samples_per_class, len(X_train)):
    y_unique_classes = np.unique(y_train[:iteration])
    if len(y_unique_classes) == len(np.unique(y_train)): 
        best_svm.fit(X_train[:iteration], y_train[:iteration])
        score = best_svm.score(X_test, y_test)
        accuracies.append(score * 100)  

plt.plot(range(min_samples_per_class, len(X_train)), accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('Convergence Graph of Best SVM for Wine Dataset')
plt.show()

results_df.to_csv('wine_svm_optimization_results.csv', index=False)
