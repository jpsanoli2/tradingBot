"""Analyze historical trade performance and project new config results."""
from data.database import db
from data.models import Trade, BalanceHistory
from sqlalchemy import func

session = db.get_session()

# Total trades
total = session.query(Trade).count()
closed = session.query(Trade).filter(Trade.status == 'closed').count()
open_t = session.query(Trade).filter(Trade.status == 'open').count()

print("=== TRADE HISTORY ===")
print(f"Total trades: {total}")
print(f"Closed: {closed}")
print(f"Open: {open_t}")

if closed > 0:
    wins = session.query(Trade).filter(Trade.status == 'closed', Trade.pnl > 0).count()
    losses = session.query(Trade).filter(Trade.status == 'closed', Trade.pnl <= 0).count()
    win_rate = wins / closed * 100
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"Losses: {losses} ({losses/closed*100:.1f}%)")

    # PnL stats
    total_pnl = session.query(func.sum(Trade.pnl)).filter(Trade.status == 'closed').scalar() or 0
    avg_pnl = session.query(func.avg(Trade.pnl)).filter(Trade.status == 'closed').scalar() or 0
    avg_win = session.query(func.avg(Trade.pnl)).filter(Trade.status == 'closed', Trade.pnl > 0).scalar() or 0
    avg_loss = session.query(func.avg(Trade.pnl)).filter(Trade.status == 'closed', Trade.pnl <= 0).scalar() or 0
    max_win = session.query(func.max(Trade.pnl)).filter(Trade.status == 'closed').scalar() or 0
    max_loss = session.query(func.min(Trade.pnl)).filter(Trade.status == 'closed').scalar() or 0

    print("")
    print("=== PNL STATS ===")
    print(f"Total PnL: ${total_pnl:+,.2f}")
    print(f"Avg PnL per trade: ${avg_pnl:+,.2f}")
    print(f"Avg Win: ${avg_win:+,.2f}")
    print(f"Avg Loss: ${avg_loss:+,.2f}")
    print(f"Best trade: ${max_win:+,.2f}")
    print(f"Worst trade: ${max_loss:+,.2f}")

    if avg_loss != 0:
        rr = abs(avg_win / avg_loss)
        print(f"Actual R:R: {rr:.2f}")

    # Win streaks and loss streaks
    all_trades = session.query(Trade).filter(Trade.status == 'closed').order_by(Trade.exit_time.asc()).all()
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_type = None
    for t in all_trades:
        if t.pnl and t.pnl > 0:
            if current_type == 'win':
                current_streak += 1
            else:
                current_type = 'win'
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_type == 'loss':
                current_streak += 1
            else:
                current_type = 'loss'
                current_streak = 1
            max_loss_streak = max(max_loss_streak, current_streak)
    
    print(f"Max win streak: {max_win_streak}")
    print(f"Max loss streak: {max_loss_streak}")

# Last trades
print("")
print("=== LAST 20 TRADES ===")
last_trades = session.query(Trade).filter(Trade.status == 'closed').order_by(Trade.exit_time.desc()).limit(20).all()
for t in last_trades:
    emoji = "W" if t.pnl and t.pnl > 0 else "L"
    pnl_str = f"${t.pnl:+,.2f}" if t.pnl else "N/A"
    pnl_pct_str = f"{t.pnl_pct:+.2f}%" if t.pnl_pct else ""
    entry = f"${t.entry_price:,.0f}" if t.entry_price else "?"
    exit_p = f"${t.exit_price:,.0f}" if t.exit_price else "?"
    sl = f"${t.stop_loss:,.0f}" if t.stop_loss else "?"
    tp = f"${t.take_profit:,.0f}" if t.take_profit else "?"
    exit_t = str(t.exit_time)[:16] if t.exit_time else "?"
    notes = (t.notes or "")[-50:]
    print(f"  [{emoji}] {t.side:4s} @ {entry} -> {exit_p} | SL:{sl} TP:{tp} | PnL: {pnl_str} ({pnl_pct_str}) | {exit_t} | {notes}")

# Balance history
print("")
print("=== BALANCE ===")
latest_bal = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.desc()).first()
if latest_bal:
    print(f"Latest balance: ${latest_bal.total_balance:,.2f}")
    print(f"Available: ${latest_bal.available_balance:,.2f}")

first_bal = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.asc()).first()
if first_bal:
    print(f"First recorded: ${first_bal.total_balance:,.2f}")
    if latest_bal:
        change = latest_bal.total_balance - first_bal.total_balance
        change_pct = (change / first_bal.total_balance) * 100 if first_bal.total_balance > 0 else 0
        print(f"Change: ${change:+,.2f} ({change_pct:+.1f}%)")

# Trades per day
if closed > 0:
    first_trade = session.query(Trade).order_by(Trade.entry_time.asc()).first()
    last_trade = session.query(Trade).order_by(Trade.exit_time.desc()).first()
    if first_trade and last_trade and last_trade.exit_time and first_trade.entry_time:
        days = max((last_trade.exit_time - first_trade.entry_time).days, 1)
        print("")
        print("=== FREQUENCY ===")
        print(f"Trading period: {days} days")
        print(f"Trades/day avg: {closed/days:.1f}")

        # SL vs TP hit analysis
        sl_hits = 0
        tp_hits = 0
        for t in all_trades:
            notes_lower = (t.notes or "").lower()
            if "stop-loss" in notes_lower or "stop_loss" in notes_lower:
                sl_hits += 1
            elif "take-profit" in notes_lower or "take_profit" in notes_lower:
                tp_hits += 1
        
        print("")
        print("=== EXIT REASONS ===")
        print(f"SL hits: {sl_hits}")
        print(f"TP hits: {tp_hits}")
        print(f"Other: {closed - sl_hits - tp_hits}")

session.close()
