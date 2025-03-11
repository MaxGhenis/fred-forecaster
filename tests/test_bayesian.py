import unittest
import pandas as pd
import numpy as np
import pytest
from src.forecast import fit_bayesian_model, generate_bayesian_simulations


class TestBayesianForecasting(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Create quarterly time series data (shorter for faster tests)
        dates = pd.date_range(start='2021-01-01', periods=8, freq='Q')
        # Linear trend with seasonal pattern
        values = np.array([100, 90, 120, 110, 140, 130, 160, 150])
        self.test_series = pd.Series(values, index=dates)
        self.test_df = pd.DataFrame({'Debt': self.test_series})
        self.test_df.index = pd.PeriodIndex(self.test_df.index, freq='Q-DEC')
    
    @pytest.mark.slow  # Mark as slow test to skip in quick test runs
    def test_bayesian_model_fitting(self):
        """Test that Bayesian model fits successfully"""
        # This test can be slow due to MCMC sampling
        model, idata = fit_bayesian_model(self.test_series)
        
        # Basic assertions
        self.assertIsNotNone(model)
        self.assertIsNotNone(idata)
        
        # Check that posterior contains expected variables
        expected_vars = ["sigma_level", "sigma_trend", "sigma_seasonal", "sigma_obs", 
                         "level", "trend", "seasonal"]
        for var in expected_vars:
            self.assertIn(var, idata.posterior)
    
    @pytest.mark.slow  # Mark as slow test to skip in quick test runs
    def test_bayesian_simulations(self):
        """Test that Bayesian simulations are generated correctly"""
        # Fit the model (with minimal samples for test speed)
        with unittest.mock.patch('pymc.sample', return_value=None):
            model, idata = fit_bayesian_model(self.test_series)
            
            # Mock the posterior samples
            n_data = len(self.test_series)
            mock_posterior = {
                "level": np.random.normal(size=(2, 500, n_data)),
                "trend": np.random.normal(size=(2, 500, n_data)),
                "seasonal": np.random.normal(size=(2, 500, n_data)),
                "sigma_obs": np.abs(np.random.normal(size=(2, 500)))
            }
            
            # Create mock InferenceData object
            class MockPosterior:
                def __init__(self, data):
                    self.data = data
                
                def __getitem__(self, key):
                    return self.data[key]
            
            class MockInferenceData:
                def __init__(self, posterior):
                    self.posterior = posterior
            
            mock_idata = MockInferenceData(MockPosterior(mock_posterior))
            
            # Generate simulations
            sim_array, forecast_index = generate_bayesian_simulations(
                model, 
                mock_idata, 
                self.test_df, 
                end=str(self.test_df.index[-1].year + 1) + "Q4",
                N=20  # Small number for test speed
            )
            
            # Assertions
            self.assertEqual(sim_array.shape[1], 20)  # 20 simulations
            self.assertEqual(len(forecast_index), 4)  # 4 quarters forecasted
            self.assertTrue(np.all(~np.isnan(sim_array)))  # No NaN values
            
            # Check that the forecast starts after the last data point
            self.assertTrue(forecast_index[0] > self.test_df.index[-1])


if __name__ == '__main__':
    unittest.main()