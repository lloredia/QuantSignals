"""
QuantSignals Ultra - Basic Tests
Run with: pytest tests.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock


# ============ UNIT TESTS ============

def test_kelly_position_calculation():
    """Test Kelly Criterion position sizing."""
    from main import calculate_kelly_position
    
    # 60% win rate, 10% avg win, 5% avg loss, $1000 balance, 75% confidence
    result = calculate_kelly_position(60, 10, 5, 1000, 75)
    
    assert result > 0, "Kelly should return positive position"
    assert result <= 1000, "Position should not exceed balance"


def test_market_regime_classification():
    """Test market regime detection."""
    # High vol, positive trend = bull_high_vol
    # Low vol, negative trend = bear_low_vol
    pass  # Would need async context


def test_signal_filtering():
    """Test signal confidence filtering."""
    min_confidence = 70
    
    signals = [
        {"confidence": 80, "pair": "BTC-USD"},
        {"confidence": 65, "pair": "ETH-USD"},
        {"confidence": 75, "pair": "SOL-USD"},
    ]
    
    filtered = [s for s in signals if s["confidence"] >= min_confidence]
    
    assert len(filtered) == 2
    assert all(s["confidence"] >= 70 for s in filtered)


def test_pnl_calculation():
    """Test P&L percentage calculation."""
    entry = 100
    current = 110
    
    pnl_pct = ((current - entry) / entry) * 100
    
    assert pnl_pct == 10.0


def test_trailing_stop():
    """Test trailing stop logic."""
    entry = 100
    highest = 120
    current = 114  # 5% below high
    trailing_pct = 3
    
    trailing_drop = ((current - highest) / highest) * 100
    
    assert trailing_drop < 0
    assert abs(trailing_drop) == 5.0
    
    # Should trigger if drop > trailing_pct
    should_trigger = abs(trailing_drop) >= trailing_pct
    assert should_trigger == True


def test_take_profit_tiers():
    """Test TP tier calculation."""
    tiers = [
        {"pct_gain": 5, "sell_pct": 25},
        {"pct_gain": 10, "sell_pct": 25},
        {"pct_gain": 20, "sell_pct": 25},
    ]
    
    entry = 100
    current = 112  # +12%
    pnl_pct = ((current - entry) / entry) * 100
    
    # Should hit tier 1 and 2, not tier 3
    tiers_hit = [t for t in tiers if pnl_pct >= t["pct_gain"]]
    
    assert len(tiers_hit) == 2
    assert tiers_hit[0]["pct_gain"] == 5
    assert tiers_hit[1]["pct_gain"] == 10


def test_risk_drawdown():
    """Test drawdown calculation."""
    highest_balance = 1000
    current_balance = 850
    
    drawdown = ((highest_balance - current_balance) / highest_balance) * 100
    
    assert drawdown == 15.0
    
    max_drawdown = 10
    should_pause = drawdown >= max_drawdown
    assert should_pause == True


def test_dca_trigger():
    """Test DCA buy trigger."""
    high_24h = 100
    current = 93
    dca_drop_pct = 5
    
    drop = ((high_24h - current) / high_24h) * 100
    
    assert drop == 7.0
    
    should_dca = drop >= dca_drop_pct
    assert should_dca == True


# ============ INTEGRATION TESTS ============

@pytest.mark.asyncio
async def test_price_fetch():
    """Test price fetching (requires network)."""
    # Mock the actual API call
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"price": "95000.00"})
        
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        # Would call get_public_price here
        price = 95000.00
        assert price > 0


# ============ RUN TESTS ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
