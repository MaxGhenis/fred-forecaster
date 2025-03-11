import unittest
import pandas as pd
import numpy as np
from src.forecast import fit_sarimax_model, generate_simulations


class TestForecast(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Create quarterly time series data
        dates = pd.date_range(start='2020-01-01', periods=12, freq='Q')
        values = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210])
        self.test_series = pd.Series(values, index=dates)
        self.test_df = pd.DataFrame({'Debt': self.test_series})
        self.test_df.index = pd.PeriodIndex(self.test_df.index, freq='Q-DEC')
        
    def test_fit_sarimax_model(self):
        """Test that SARIMAX model fits successfully"""
        # Fit the model
        results = fit_sarimax_model(self.test_series)
        
        # Basic assertions
        self.assertIsNotNone(results)
        self.assertTrue(hasattr(results, 'params'))
        self.assertTrue(hasattr(results, 'aic'))
        
    def test_generate_simulations(self):
        """Test that simulations are generated correctly"""
        # Fit the model
        results = fit_sarimax_model(self.test_series)
        
        # Generate simulations for 4 quarters ahead
        sim_array, forecast_index = generate_simulations(
            results, 
            self.test_df, 
            end=str(self.test_df.index[-1].year + 1) + "Q4",
            N=100
        )
        
        # Assertions
        self.assertEqual(sim_array.shape[1], 100)  # 100 simulations
        self.assertEqual(len(forecast_index), 4)   # 4 quarters forecasted
        self.assertTrue(np.all(~np.isnan(sim_array)))  # No NaN values
        
        # Check that the forecast starts after the last data point
        self.assertTrue(forecast_index[0] > self.test_df.index[-1])


if __name__ == '__main__':
    unittest.main()