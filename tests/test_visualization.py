import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import plot_forecasts, plot_drop_probabilities


class TestVisualization(unittest.TestCase):
    
    def setUp(self):
        """Create test data for visualization"""
        # Create historical data
        dates = pd.date_range(start='2020-01-01', periods=12, freq='Q')
        values = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210])
        series = pd.Series(values, index=dates)
        self.df_quarterly = pd.DataFrame({'Debt': series})
        self.df_quarterly.index = pd.PeriodIndex(self.df_quarterly.index, freq='Q-DEC')
        
        # Create simulated forecast paths
        self.n_steps = 4  # 4 quarters
        self.n_sims = 100  # 100 simulations
        
        # Create random simulations with increasing values
        np.random.seed(42)
        base_values = np.linspace(220, 280, self.n_steps)
        noise = np.random.normal(0, 10, (self.n_steps, self.n_sims))
        self.sim_array = base_values[:, np.newaxis] + noise
        
        # Create forecast index
        self.forecast_index = pd.period_range(
            start=self.df_quarterly.index[-1] + 1, 
            periods=self.n_steps, 
            freq='Q-DEC'
        )
        
        # Create some weights (not all equal)
        self.weights = np.ones(self.n_sims) / self.n_sims
        self.weights[:10] = self.weights[:10] * 2
        self.weights = self.weights / np.sum(self.weights)
        
    def test_plot_forecasts(self):
        """Test that forecast plots are created correctly"""
        # Create plot without weights
        fig1 = plot_forecasts(self.df_quarterly, self.sim_array, self.forecast_index)
        self.assertIsNotNone(fig1)
        
        # Create plot with weights
        fig2 = plot_forecasts(self.df_quarterly, self.sim_array, self.forecast_index, self.weights)
        self.assertIsNotNone(fig2)
        
        # Clean up
        plt.close('all')
        
    def test_plot_drop_probabilities(self):
        """Test that drop probability plots are created correctly"""
        # Create plot without weights
        fig1 = plot_drop_probabilities(self.sim_array, self.forecast_index)
        self.assertIsNotNone(fig1)
        
        # Create plot with weights
        fig2 = plot_drop_probabilities(self.sim_array, self.forecast_index, self.weights)
        self.assertIsNotNone(fig2)
        
        # Clean up
        plt.close('all')


if __name__ == '__main__':
    unittest.main()