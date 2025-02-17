import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"item_id": ["item1", "item2", "item3"], "drop_rate": [0.1, 0.05, 0.01], "received": [1, 0, 0]})


@pytest.fixture
def test_data_path(tmp_path):
    """Create a temporary CSV file with test data."""
    data = pd.DataFrame({"item_id": ["item1", "item2"], "drop_rate": [0.1, 0.05], "received": [1, 0]})
    file_path = tmp_path / "test_data.csv"
    data.to_csv(file_path, index=False)
    return file_path
