
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from strategies.base_strategy import TradingSignal, SignalType
from risk.risk_manager import RiskManager
from config import settings

def test_signal_actionable():
    print("\n--- Testing Signal Actionability ---")
    print(f"Current MIN_SIGNAL_STRENGTH: {settings.trading.MIN_SIGNAL_STRENGTH}")
    
    # 0.4 should be actionable if MIN_SIGNAL_STRENGTH is 0.3
    signal_weak = TradingSignal(
        action=SignalType.BUY,
        strength=0.4,
        strategy_name="Test",
        price=50000.0
    )
    
    print(f"Signal strength 0.4 is_actionable: {signal_weak.is_actionable}")
    assert signal_weak.is_actionable == True, "Signal should be actionable"

    # 0.2 should NOT be actionable if MIN_SIGNAL_STRENGTH is 0.3
    signal_very_weak = TradingSignal(
        action=SignalType.BUY,
        strength=0.2,
        strategy_name="Test",
        price=50000.0
    )
    print(f"Signal strength 0.2 is_actionable: {signal_very_weak.is_actionable}")
    assert signal_very_weak.is_actionable == False, "Signal should NOT be actionable"

def test_redundancy_protection():
    print("\n--- Testing Redundancy Protection ---")
    risk = RiskManager()
    
    signal = TradingSignal(
        action=SignalType.BUY,
        strength=0.8,
        strategy_name="Test",
        price=50000.0
    )
    
    balance = 1000.0
    price = 50000.0
    
    # Case 1: No open positions
    result1 = risk.evaluate_signal(signal, balance, price, open_directions=[])
    print(f"No open positions result: Approved={result1['approved']}, Reason={result1['reason']}")
    assert result1['approved'] == True
    
    # Case 2: Open position in DIFFERENT direction (SELL)
    result2 = risk.evaluate_signal(signal, balance, price, open_directions=["sell"])
    print(f"Different direction (SELL) open: Approved={result2['approved']}, Reason={result2['reason']}")
    assert result2['approved'] == True
    
    # Case 3: Open position in SAME direction (BUY)
    result3 = risk.evaluate_signal(signal, balance, price, open_directions=["buy"])
    print(f"Same direction (BUY) open: Approved={result3['approved']}, Reason={result3['reason']}")
    assert result3['approved'] == False
    assert "Already have an open BUY position" in result3['reason']

if __name__ == "__main__":
    try:
        test_signal_actionable()
        test_redundancy_protection()
        print("\nVerification SUCCESSFUL")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        sys.exit(1)
