"""
Automated tests for ML assignment
"""

import pytest
import numpy as np
from starter_code.ml_basics import (
    normalize_features,
    handle_missing_values,
    split_data,
    fit_linear_regression,
    predict_linear,
    fit_logistic_regression,
    predict_class,
    calculate_mse,
    calculate_accuracy
)


class TestDataPreprocessing:
    
    def test_normalize_features(self):
        """Test feature normalization (10 points)"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_features(X)
        
        assert result is not None, "Function returns None"
        assert result.shape == X.shape, "Output shape doesn't match input"
        assert np.allclose(result.min(), 0), "Minimum should be 0"
        assert np.allclose(result.max(), 1), "Maximum should be 1"
        assert np.allclose(result[0], [0, 0]), "First row should be [0, 0]"
        assert np.allclose(result[-1], [1, 1]), "Last row should be [1, 1]"
    
    def test_handle_missing_values_mean(self):
        """Test missing value handling with mean (5 points)"""
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        result = handle_missing_values(X, strategy='mean')
        
        assert result is not None, "Function returns None"
        assert not np.isnan(result).any(), "Result still contains NaN values"
        assert result.shape == X.shape, "Shape changed"
        assert np.allclose(result[1, 0], 3.0), "Mean imputation incorrect"
    
    def test_handle_missing_values_median(self):
        """Test missing value handling with median (5 points)"""
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        result = handle_missing_values(X, strategy='median')
        
        assert result is not None, "Function returns None"
        assert not np.isnan(result).any(), "Result still contains NaN values"


class TestTrainTestSplit:
    
    def test_split_data(self):
        """Test train-test split (15 points)"""
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert X_train.shape[0] == 80, "Training set size incorrect"
        assert X_test.shape[0] == 20, "Test set size incorrect"
        assert y_train.shape[0] == 80, "Training labels size incorrect"
        assert y_test.shape[0] == 20, "Test labels size incorrect"
        assert X_train.shape[1] == X.shape[1], "Feature dimensions changed"


class TestLinearRegression:
    
    def test_fit_linear_regression(self):
        """Test linear regression training (15 points)"""
        X_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 4, 6, 8, 10])
        
        model = fit_linear_regression(X_train, y_train)
        
        assert model is not None, "Function returns None"
        assert hasattr(model, 'predict'), "Model doesn't have predict method"
        
        # Test if model learned the pattern (y = 2x)
        predictions = model.predict([[6]])
        assert np.abs(predictions[0] - 12) < 0.5, "Model didn't learn the pattern"
    
    def test_predict_linear(self):
        """Test linear regression prediction (10 points)"""
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([1, 2, 3])
        
        model = fit_linear_regression(X_train, y_train)
        X_test = np.array([[4], [5]])
        predictions = predict_linear(model, X_test)
        
        assert predictions is not None, "Function returns None"
        assert len(predictions) == 2, "Wrong number of predictions"
        assert predictions.shape == (2,), "Wrong prediction shape"


class TestLogisticRegression:
    
    def test_fit_logistic_regression(self):
        """Test logistic regression training (15 points)"""
        np.random.seed(42)
        X_train = np.random.rand(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)
        
        model = fit_logistic_regression(X_train, y_train)
        
        assert model is not None, "Function returns None"
        assert hasattr(model, 'predict'), "Model doesn't have predict method"
        
        # Test prediction
        predictions = model.predict(X_train)
        accuracy = (predictions == y_train).mean()
        assert accuracy > 0.7, "Model accuracy too low"
    
    def test_predict_class(self):
        """Test classification prediction (10 points)"""
        np.random.seed(42)
        X_train = np.random.rand(100, 2)
        y_train = (X_train[:, 0] > 0.5).astype(int)
        
        model = fit_logistic_regression(X_train, y_train)
        X_test = np.array([[0.8, 0.2], [0.2, 0.8]])
        predictions = predict_class(model, X_test)
        
        assert predictions is not None, "Function returns None"
        assert len(predictions) == 2, "Wrong number of predictions"
        assert all(p in [0, 1] for p in predictions), "Predictions not binary"


class TestEvaluation:
    
    def test_calculate_mse(self):
        """Test MSE calculation (8 points)"""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        mse = calculate_mse(y_true, y_pred)
        
        assert mse is not None, "Function returns None"
        assert isinstance(mse, (int, float)), "MSE should be a number"
        assert np.allclose(mse, 0.375), "MSE calculation incorrect"
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation (7 points)"""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        
        accuracy = calculate_accuracy(y_true, y_pred)
        
        assert accuracy is not None, "Function returns None"
        assert isinstance(accuracy, (int, float)), "Accuracy should be a number"
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert np.allclose(accuracy, 0.8), "Accuracy calculation incorrect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])