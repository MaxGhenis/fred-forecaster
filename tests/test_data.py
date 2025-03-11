import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from src.data import fetch_fred_data


class TestData(unittest.TestCase):
    
    @patch('src.data.Fred')
    def test_fetch_fred_data(self, mock_fred):
        """Test that fetch_fred_data correctly processes FRED data"""
        # Setup mock
        os.environ['FRED_API_KEY'] = 'test_key'
        
        # Create a mock time series (in millions as returned by FRED)
        mock_series = pd.Series(
            [100000000, 200000000, 300000000, 400000000],  # 100-400 million
            index=pd.date_range('2022-01-01', periods=4, freq='QE')
        )
        mock_series = mock_series.rename("Debt_millions")
        
        # Configure the mock
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred.return_value = mock_fred_instance
        
        # Call the function
        result = fetch_fred_data('TEST')
        
        # Assertions
        self.assertEqual(len(result), 4)
        self.assertTrue('Debt' in result.columns)
        self.assertEqual(result['Debt'].iloc[0], 100)  # 100 million / 1e6 = 100 trillion
        self.assertEqual(result['Debt'].iloc[-1], 400)
        
        # Verify the mock was called correctly
        mock_fred.assert_called_once_with(api_key='test_key')
        mock_fred_instance.get_series.assert_called_once_with('TEST')
        
    def test_fetch_fred_data_no_api_key(self):
        """Test that fetch_fred_data raises an error when no API key is set"""
        # Remove the API key from environment
        if 'FRED_API_KEY' in os.environ:
            del os.environ['FRED_API_KEY']
            
        # Assert that calling the function raises a ValueError
        with self.assertRaises(ValueError):
            fetch_fred_data('TEST')


if __name__ == '__main__':
    unittest.main()