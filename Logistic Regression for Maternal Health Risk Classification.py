import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    target_column = 'RiskLevel'  

    # converting risk levels to binary (0 for low risk, 1 for moderate/high risk)
    df['binary_risk'] = df[target_column].apply(lambda x: 0 if x.lower() == 'low risk' else 1)
    
    # extracting features (all columns except the risk level column)
    feature_columns = [col for col in df.columns if col != target_column and col != 'binary_risk']
    X = df[feature_columns].values
    y = df['binary_risk'].values
    
    # bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    return X, y

def sigmoid(z):
    z = np.clip(z, -500, 500) # clipped z in order to avoid overflow
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)  # initial weights are zeros

    iterations = []
    accuracies = []
    
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (y - h)) / m
        theta += learning_rate * gradient
        
        if i % 10 == 0 or i == num_iterations - 1: # calculate accuracy every 10 iterations to save computation
            iterations.append(i)
            y_pred = predict(X, theta)
            accuracy = np.mean(y_pred == y)
            accuracies.append(accuracy)
    
    return theta, iterations, accuracies

def predict(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return (h >= 0.5).astype(int)

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    matrix = np.array([[TP, FP], [FN, TN]])
    return matrix





def main():
    train_path = 'train_data.csv'
    val_path = 'val_data.csv'
    test_path = 'test_data.csv'
    
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)
    
    print(f"Training set: X shape = {X_train.shape}, y shape = {y_train.shape}")
    print(f"Validation set: X shape = {X_val.shape}, y shape = {y_val.shape}")
    print(f"Test set: X shape = {X_test.shape}, y shape = {y_test.shape}")
    
    # normalizing features
    mean = np.mean(X_train[:, 1:], axis=0)
    std = np.std(X_train[:, 1:], axis=0)
    
    
    std = np.where(std == 0, 1e-5, std) # to avoid division by zero
    
    X_train[:, 1:] = (X_train[:, 1:] - mean) / std
    X_val[:, 1:] = (X_val[:, 1:] - mean) / std
    X_test[:, 1:] = (X_test[:, 1:] - mean) / std
    
    learning_rates = [1e-3, 1e-2, 1e-1, 1, 10]
    num_iterations = 1000
    
    plt.figure(figsize=(12, 8))
    
    best_accuracy = 0
    best_lr = 0
    best_theta = None
    
    val_accuracies_by_lr = {}
    
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        theta, iterations, train_accuracies = logistic_regression(X_train, y_train, lr, num_iterations)
        y_val_pred = predict(X_val, theta)
        val_accuracy = np.mean(y_val_pred == y_val)
        print(f"Validation accuracy with learning rate {lr}: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_lr = lr
            best_theta = theta
        
        val_accuracies = []
        for i in range(0, num_iterations, 10):
            theta_i, _, _ = logistic_regression(X_train, y_train, lr, i+1)
            y_val_pred_i = predict(X_val, theta_i)
            val_accuracy_i = np.mean(y_val_pred_i == y_val)
            val_accuracies.append(val_accuracy_i)
        
        val_accuracies_by_lr[lr] = val_accuracies
    
    plt.figure(figsize=(12, 8))
    iterations_to_plot = list(range(0, num_iterations, 10))
    
    for lr, accuracies in val_accuracies_by_lr.items():
        plt.plot(iterations_to_plot, accuracies, label=f'LR = {lr}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nBest learning rate: {best_lr} with validation accuracy: {best_accuracy:.4f}")
    
    best_theta, _, _ = logistic_regression(X_train, y_train, best_lr, num_iterations) # train with the best learning rate
    
    y_test_pred = predict(X_test, best_theta)
    test_accuracy = np.mean(y_test_pred == y_test)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    print(f"\nTest accuracy with best learning rate ({best_lr}): {test_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nConfusion Matrix Interpretation:")
    print(f"True Negatives: {conf_matrix[0, 0]}")
    print(f"False Positives: {conf_matrix[0, 1]}")
    print(f"False Negatives: {conf_matrix[1, 0]}")
    print(f"True Positives: {conf_matrix[1, 1]}")

if __name__ == "__main__":
    main()