import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

from config import settings
from data.database import db
from data.models import OHLCV, Trade, Signal, BalanceHistory, PriceTick
from core.engine import TradingEngine


app = FastAPI(title="AI Trading Bot Dashboard")

# Global engine state
class BotState:
    def __init__(self):
        self.engine: Optional[TradingEngine] = None
        self.scheduler: Optional[BackgroundScheduler] = None
        self.running = False
        self.lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.total_realized_pnl = 0.0
        self.base_balance = settings.trading.INITIAL_CAPITAL

state = BotState()

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Templates setup
templates_path = settings.BASE_DIR / "dashboard" / "templates"
templates_path.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_path))


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/price")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/status")
async def get_status():
    """Get overall bot status and recent metrics."""
    session = db.get_session()
    try:
        # Get latest balance
        balance = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.desc()).first()
        
        # Get trade statistics
        total_trades = session.query(Trade).count()
        closed_trades = session.query(Trade).filter(Trade.status == "closed").all()
        
        # Metrics calculations
        gross_profit = sum(t.pnl for t in closed_trades if (t.pnl or 0) > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if (t.pnl or 0) < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 1.0)
        
        # Max Drawdown calculation from balance history
        all_balances = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.asc()).all()
        max_dd = 0.0
        if all_balances:
            peak = all_balances[0].total_balance
            for b in all_balances:
                if b.total_balance > peak:
                    peak = b.total_balance
                dd = (peak - b.total_balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        
        total_pnl = sum(t.pnl or 0 for t in closed_trades)
        winning_trades = len([t for t in closed_trades if (t.pnl or 0) > 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        
        engine_status = state.engine.get_status() if state.engine else {}
        
        return {
            "running": state.running,
            "pair": settings.trading.PAIR,
            "mode": settings.trading.MODE,
            "market_type": getattr(settings.exchange, "MARKET_TYPE", "spot"),
            "leverage": getattr(settings.exchange, "LEVERAGE", 1),
            "current_balance": balance.total_balance if balance else settings.trading.INITIAL_CAPITAL,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown": round(max_dd, 2),
            "trades_count": total_trades,
            "open_positions": session.query(Trade).filter(Trade.status == "open").count(),
            "ai_ready": engine_status.get("ai_ready", False),
            "last_signal": engine_status.get("last_signal", {"action": None, "strength": 0})
        }
    finally:
        session.close()


@app.post("/api/start")
async def start_bot():
    """Start the trading engine."""
    with state.lock:
        if state.running:
            return {"status": "already_running"}
        
        try:
            if not state.engine:
                state.engine = TradingEngine()
            
            state.engine.start()
            state.loop = asyncio.get_event_loop()
            
            # Setup Scheduler
            # Timeframe to seconds mapping
            TIMEFRAME_SECONDS = {
                "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
            }
            interval_seconds = TIMEFRAME_SECONDS.get(settings.trading.TIMEFRAME, 3600)
            
            state.scheduler = BackgroundScheduler()
            
            # 1. Main Trading Cycle (e.g., every 1h)
            state.scheduler.add_job(
                state.engine.run_cycle,
                "interval",
                seconds=interval_seconds,
                id="trading_cycle",
                max_instances=1,
            )

            # Initial static data for cached metrics
            session = db.get_session()
            try:
                closed_trades = session.query(Trade).filter(Trade.status == "closed").all()
                state.total_realized_pnl = sum(t.pnl or 0 for t in closed_trades)
                
                balance_record = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.desc()).first()
                state.base_balance = balance_record.total_balance if balance_record else settings.trading.INITIAL_CAPITAL
            finally:
                session.close()

            # 2. Market Data Watcher (every 1m) - Updates the current candle for the dashboard
            def update_market_data():
                try:
                    state.engine.data_collector.fetch_and_store(limit=2)
                    logger.debug("Market data refreshed for dashboard")
                except Exception as e:
                    logger.error(f"Watcher failed: {e}")

            state.scheduler.add_job(
                update_market_data,
                "interval",
                minutes=1,
                id="market_watcher"
            )
            # 3. Ticker Watcher (every 2s) - Saves every price and broadcasts to WebSocket
            def update_ticker():
                try:
                    ticker = state.engine.data_collector.fetch_and_store_ticker()
                    current_price = ticker["last"]
                    
                    metrics = state.engine.get_unrealized_metrics(current_price)
                    
                    msg = {
                        "type": "price_update",
                        "price": current_price,
                        "timestamp": ticker["timestamp"].isoformat(),
                        "metrics": {
                            "total_pnl": round(state.total_realized_pnl + metrics["total_unrealized_pnl"], 2),
                            "current_balance": round(state.base_balance + metrics["total_unrealized_pnl"], 2),
                            "unrealized_pnl": metrics["total_unrealized_pnl"],
                            "open_trades": metrics["trades"]
                        }
                    }
                    
                    if state.loop and state.loop.is_running():
                        asyncio.run_coroutine_threadsafe(manager.broadcast(msg), state.loop)
                    
                    logger.info(f"Broadcasted ticker: {current_price} | Open trades: {len(metrics['trades'])}")
                except Exception as e:
                    logger.error(f"Ticker watcher failed: {e}")

            state.scheduler.add_job(
                update_ticker,
                "interval",
                seconds=2,
                id="ticker_watcher"
            )
            
            # Run first cycle in background
            threading.Thread(target=state.engine.run_cycle).start()
            
            state.scheduler.start()
            state.running = True
            logger.info("Bot started via Dashboard")
            return {"status": "started"}
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_bot():
    """Stop the trading engine."""
    with state.lock:
        if not state.running:
            return {"status": "already_stopped"}
        
        try:
            if state.scheduler:
                state.scheduler.shutdown()
            if state.engine:
                state.engine.stop()
            
            state.running = False
            logger.info("Bot stopped via Dashboard")
            return {"status": "stopped"}
        except Exception as e:
            logger.error(f"Failed to stop bot: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ohlcv")
async def get_ohlcv_data(limit: int = 100):
    """Get recent candle data for charts."""
    session = db.get_session()
    try:
        candles = session.query(OHLCV).order_by(OHLCV.timestamp.desc()).limit(limit).all()
        data = [
            {
                "time": int(c.timestamp.timestamp()),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in reversed(candles)
        ]
        return data
    finally:
        session.close()


@app.get("/api/trades")
async def get_trades(limit: int = 20):
    """Get recent trades."""
    session = db.get_session()
    try:
        trades = session.query(Trade).order_by(Trade.entry_time.desc()).limit(limit).all()
        return [
            {
                "id": t.id,
                "pair": t.pair,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat(),
                "pnl": t.pnl,
                "status": t.status,
                "take_profit": t.take_profit,
                "stop_loss": t.stop_loss,
            }
            for t in trades
        ]
    finally:
        session.close()


@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """Get the latest log entries."""
    log_file = settings.BASE_DIR / "logs" / "trading_bot.log"
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Clean ANSI escape codes from lines
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_lines = [ansi_escape.sub('', line).strip() for line in lines]
            return clean_lines[-limit:]
    except Exception:
        return []
@app.get("/api/price-history")
async def get_price_history(limit: int = 200):
    """Get recent price ticks."""
    session = db.get_session()
    try:
        ticks = session.query(PriceTick).order_by(PriceTick.timestamp.desc()).limit(limit).all()
        return [
            {
                "time": int(t.timestamp.timestamp()),
                "price": t.price
            }
            for t in reversed(ticks)
        ]
    finally:
        session.close()


@app.get("/api/equity")
async def get_equity_curve():
    """Get balance history for equity curve chart."""
    session = db.get_session()
    try:
        history = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.asc()).all()
        return [
            {
                "time": int(h.timestamp.timestamp()),
                "balance": h.total_balance
            }
            for h in history
        ]
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=True)
