import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from sbi_access import get_sbi_holdings 
from dotenv import load_dotenv
import os

load_dotenv()
def test_get_sbi_holdings():
    username = os.getenv("SBI_USERNAME")
    password = os.getenv("SBI_PASSWORD")
    holdings = get_sbi_holdings(username, password)

    assert isinstance(holdings, list)
    assert len(holdings) > 0
    assert all(isinstance(symbol, str) for symbol in holdings)
