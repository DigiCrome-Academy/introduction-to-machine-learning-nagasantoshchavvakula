"""
Introduction to Machine Learning - Starter Code
Complete all the functions below according to their docstrings.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


def normalize_features(X):
    """
    Normalize features to range [0, 1] using min-max scaling.
    
    Formula: X_normalized = (X - X_min) / (X_max - X_min)
    
    Args:
        X (numpy.ndarray): Input features, shape (n_samples, n_features)
    
    Returns:
        numpy.ndarray: Normalized features with same shape as input
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> normalize_features(X)
        array([[0.  , 0.  ],
               [0.5 , 0.5 ],
               [1.  , 1.  ]])
    """
    # TODO: Implement min-max normalization
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized
    # pass


def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values (NaN) in the data.
    
    Args:
        X (numpy.ndarray): Input data with possible NaN values
        strategy (str): Strategy to use - 'mean', 'median', or 'zero'
    
    Returns:
        numpy.ndarray: Data with missing values filled
    
    Example:
        >>> X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        >>> handle_missing_values(X, strategy='mean')
        array([[1., 2.],
               [3., 4.],
               [5., 3.]])
    """
    # TODO: Implement missing value handling
    X_filled = X.copy()
    for i in range(X.shape[1]):
        column = X[:, i]
        nan_mask = np.isnan(column)
        if strategy == 'mean':
            mean_value = np.nanmean(column)
            column[nan_mask] = mean_value
        elif strategy == 'median':
            median_value = np.nanmedian(column)
            column[nan_mask] = median_value
        elif strategy == 'zero':
            column[nan_mask] = 0
        X_filled[:, i] = column
    return X_filled
    # pass


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Target variable
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    
    Hint: Use sklearn.model_selection.train_test_split
    """
    # TODO: Implement train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    # pass


def fit_linear_regression(X_train, y_train):
    """
    Train a linear regression model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
    
    Returns:
        sklearn.linear_model.LinearRegression: Trained model
    
    Hint: Use sklearn.linear_model.LinearRegression
    """
    # TODO: Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
    # pass


def predict_linear(model, X_test):
    """
    Make predictions using a trained linear regression model.
    
    Args:
        model: Trained linear regression model
        X_test (numpy.ndarray): Test features
    
    Returns:
        numpy.ndarray: Predictions
    """
    # TODO: Use the model to make predictions
    LinearRegression_predictions = model.predict(X_test)
    return LinearRegression_predictions
    # pass


def fit_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model for classification.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
    
    Returns:
        sklearn.linear_model.LogisticRegression: Trained model
    
    Hint: Use sklearn.linear_model.LogisticRegression with max_iter=1000
    """
    # TODO: Create and train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model
    # pass


def predict_class(model, X_test):
    """
    Predict class labels using a trained classification model.
    
    Args:
        model: Trained classification model
        X_test (numpy.ndarray): Test features
    
    Returns:
        numpy.ndarray: Predicted class labels
    """
    # TODO: Use the model to predict class labels
    LogisticRegression_predictions = model.predict(X_test)
    return LogisticRegression_predictions
    # pass


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
    
    Returns:
        float: Mean squared error
    
    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> calculate_mse(y_true, y_pred)
        0.375
    """
    # TODO: Calculate and return MSE
    mse = np.mean((y_true - y_pred) ** 2)
    return mse
    # pass


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Formula: Accuracy = (Number of correct predictions) / (Total predictions)
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
    
    Returns:
        float: Accuracy score (between 0 and 1)
    
    Example:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> calculate_accuracy(y_true, y_pred)
        0.8
    """
    # TODO: Calculate and return accuracy
    accuracy = np.mean(y_true == y_pred)
    return accuracy
    # pass