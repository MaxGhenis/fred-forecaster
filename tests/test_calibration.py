import unittest
import pandas as pd
import numpy as np
from src.calibration import calibrate_simulations


class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        """Create test data for calibration"""
        # Create simulated forecast paths
        self.n_steps = 8  # 8 quarters, covering 2 years (including Q4 of each year)
        self.n_sims = 100  # 100 simulations
        
        # Create random simulations with increasing values that match CBO targets in Q4
        np.random.seed(42)
        
        # Include 2024 and 2025 in the forecasts - these exist in the CBO targets
        self.forecast_index = pd.period_range(start='2024Q1', periods=self.n_steps, freq='Q-DEC')
        
        # Create base values ensuring Q4 values are close to CBO targets
        # CBO targets: 2024: 35.230, 2025: 37.209
        base_values = np.array([
            32.0,    # 2024Q1
            33.0,    # 2024Q2
            34.0,    # 2024Q3
            35.23,   # 2024Q4 - CBO target
            36.0,    # 2025Q1
            36.5,    # 2025Q2 
            37.0,    # 2025Q3
            37.21    # 2025Q4 - CBO target
        ])
        
        # Add noise, but keep it small enough not to cause calibration problems
        noise = np.random.normal(0, 1.0, (self.n_steps, self.n_sims))
        self.sim_array = base_values[:, np.newaxis] + noise
        
    def test_calibration(self):
        """Test that calibration produces valid weights"""
        # Run calibration
        weights = calibrate_simulations(self.sim_array, self.forecast_index)
        
        # Assertions
        self.assertEqual(len(weights), self.n_sims)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)  # Weights sum to 1
        self.assertTrue(np.all(weights >= 0))  # All weights non-negative
        
        # Test with invalid forecast index (future years should match CBO forecasts)
        with self.assertRaises(ValueError):
            bad_index = pd.period_range(start='2050Q1', periods=4, freq='Q-DEC')
            calibrate_simulations(self.sim_array, bad_index)


if __name__ == '__main__':
    unittest.main()