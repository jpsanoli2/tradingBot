import sqlite3
import json

db_path = r'c:\Users\Juampa\Desktop\tradingBot\data\trading_bot.db'

def get_stats():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # General stats
    cursor.execute('SELECT COUNT(*), SUM(pnl), SUM(pnl_pct) FROM trades WHERE status = "closed"')
    total_trades, total_pnl, total_pnl_pct = cursor.fetchone()
    
    if total_trades == 0:
        return "No hay operaciones cerradas todavía."
    
    # Wins vs Losses
    cursor.execute('SELECT COUNT(*) FROM trades WHERE status = "closed" AND pnl > 0')
    wins = cursor.fetchone()[0]
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    
    # Balance
    cursor.execute('SELECT total_balance FROM balance_history ORDER BY timestamp DESC LIMIT 1')
    current_balance = cursor.fetchone()[0]
    
    # Recent trades
    cursor.execute('SELECT side, pair, entry_price, exit_price, pnl, pnl_pct, exit_time FROM trades WHERE status = "closed" ORDER BY exit_time DESC LIMIT 5')
    recent = cursor.fetchall()
    
    stats = {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "current_balance": current_balance,
        "recent": recent
    }
    conn.close()
    return stats

res = get_stats()
if isinstance(res, str):
    print(res)
else:
    print(json.dumps(res, indent=4))
