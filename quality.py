import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('winequality-red.csv', delimiter=';')
print("First few rows of the dataset:")
print(data.head()) 
print("Columns in the DataFrame:", data.columns) 
X = data.drop('quality', axis=1)
y = data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
results = []
accuracies_per_sample = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=i)
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf', 'poly']}
    svm = SVC()
    grid = GridSearchCV(svm, param_grid, refit=True, verbose=0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Sample': i + 1, 'Best Parameters': grid.best_params_, 'Accuracy': accuracy})
    accuracies_per_sample.append(accuracy)
results_df = pd.DataFrame(results)
print("\nResults:")
print(results_df)
for result in results:
    print(f"Sample {result['Sample']}: Best Parameters: {result['Best Parameters']}, Accuracy: {result['Accuracy']:.2f}")
results_df.to_csv('svm_results.csv', index=False)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), accuracies_per_sample, marker='o')
plt.title('SVM Accuracy for Each Sample')
plt.xlabel('Sample Number')
plt.ylabel('Accuracy')
plt.xticks(range(1, 11))
plt.grid()
plt.show()
max_accuracy_index = np.argmax(accuracies_per_sample)
print(f"Sample with maximum accuracy: Sample {max_accuracy_index + 1}, Accuracy: {accuracies_per_sample[max_accuracy_index]:.2f}")
