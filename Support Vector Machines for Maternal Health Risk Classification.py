import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


# converting the risk level to binary (0 for low risk, 1 for moderate/high risk), similar to q2
def convert_to_binary(risk_level):
    return 0 if risk_level == 'low risk' else 1

train_data['binary_risk'] = train_data['RiskLevel'].apply(convert_to_binary)
test_data['binary_risk'] = test_data['RiskLevel'].apply(convert_to_binary)
y_train = train_data['binary_risk'].values
y_test = test_data['binary_risk'].values

feature_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
X_train = train_data[feature_columns].values
X_test = test_data[feature_columns].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def k_fold_cross_validation(X, y, k=5):

    indices = np.random.permutation(len(X))    # shuffling the data
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    fold_size = len(X) // k
    folds_X = []
    folds_y = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(X)
        folds_X.append(X_shuffled[start_idx:end_idx])
        folds_y.append(y_shuffled[start_idx:end_idx])
    
    return folds_X, folds_y


np.random.seed(0)  # this line is for the reproducibility of results in different runs, the value 0 does not hold any mathematical significance in creating reproducibility
folds_X, folds_y = k_fold_cross_validation(X_train, y_train, k=5)
C_values = [0.001, 0.01, 0.1, 1, 10]
cv_results = {C: [] for C in C_values}



print("Performing 5-fold Cross-Validation:")
for C in C_values:
    print(f"Training with C = {C}")
    fold_accuracies = []
    
    for i in range(5):
        X_val = folds_X[i]
        y_val = folds_y[i]
        
        # new training sets from all other folds
        X_cv_train = np.vstack([folds_X[j] for j in range(5) if j != i])
        y_cv_train = np.concatenate([folds_y[j] for j in range(5) if j != i])
        
        svm = SVC(C=C, kernel='linear', random_state=0)
        svm.fit(X_cv_train, y_cv_train)

        y_val_pred = svm.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        fold_accuracies.append(accuracy)
    
    mean_accuracy = np.mean(fold_accuracies)
    cv_results[C] = fold_accuracies
    print(f"Mean CV Accuracy: {mean_accuracy:.4f}")

mean_accuracies = {C: np.mean(accuracies) for C, accuracies in cv_results.items()}
best_C = max(mean_accuracies, key=mean_accuracies.get)
print(f"\nBest C value: {best_C} with mean CV accuracy: {mean_accuracies[best_C]:.4f}")

# cross-validation results for display
cv_table = pd.DataFrame({
    'C': list(cv_results.keys()),
    'Fold 1': [cv_results[C][0] for C in cv_results],
    'Fold 2': [cv_results[C][1] for C in cv_results],
    'Fold 3': [cv_results[C][2] for C in cv_results],
    'Fold 4': [cv_results[C][3] for C in cv_results],
    'Fold 5': [cv_results[C][4] for C in cv_results],
    'Mean': [np.mean(cv_results[C]) for C in cv_results]
})
print("\nCross-Validation Results:")
print(cv_table.to_string(index=False))

# the final model using the best C value
final_model = SVC(C=best_C, kernel='linear', random_state=0)
final_model.fit(X_train, y_train)

y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nTest Results with Best Model (C = {})".format(best_C))
print("Test Accuracy: {:.4f}".format(test_accuracy))


cm = confusion_matrix(y_test, y_test_pred) # confusion_matrix version places predictions on horizontal axis, printing manually
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix:")
print(f"True Negatives (TN): {tn}  |  False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}  |  True Positives (TP): {tp}")
print("           |  Actual Positive  |  Actual Negative  |")
print("-----------|----------------------|----------------------|")
print(f"Predicted Positive |         {tp}          |         {fp}          |")
print(f"Predicted Negative |         {fn}          |         {tn}          |")
print("-----------|----------------------|----------------------|")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

print("\nPrecision: {:.4f}".format(test_precision))
print("Recall: {:.4f}".format(test_recall))
print("F1 Score: {:.4f}".format(test_f1))