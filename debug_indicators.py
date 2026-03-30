
import pandas as pd
from analysis.indicators import TechnicalIndicators
from exchange.connector import ExchangeConnector
from config import settings

def main():
    try:
        exchange = ExchangeConnector()
        df = exchange.get_ohlcv(settings.trading.PAIR, settings.trading.TIMEFRAME, limit=100)
        
        indicators = TechnicalIndicators()
        df_with_indicators = indicators.calculate_all(df)
        
        print(f"Columns: {df_with_indicators.columns.tolist()}")
        print("\nLast 5 rows with ATR and Bollinger Bands:")
        subset = df_with_indicators.tail(5)
        cols_to_show = ["close", "atr", "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"]
        cols_available = [c for c in cols_to_show if c in df_with_indicators.columns]
        print(subset[cols_available])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
