"""
Kotegawa Trading Strategy Monitor - FastAPI Backend (yfinance Version)
Completely free - no API keys needed. Works on iPhone + web.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kotegawa Trading Monitor",
    description="Free trading strategy monitor for Indian stocks"
)

# Enable CORS for iPhone/web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Models ====================

class MonitorConfig(BaseModel):
    stocks: List[str]
    investment_per_signal: float = 10000
    check_interval: int = 3600  # seconds (default 1 hour)

class SignalLog(BaseModel):
    stock: str
    signal_type: str  # "BUY"
    entry_price: float
    entry_time: str
    investment: float
    current_price: Optional[float] = None
    current_pnl: Optional[float] = None
    current_pnl_percent: Optional[float] = None
    status: str = "ACTIVE"  # ACTIVE, CLOSED

class IndicatorData(BaseModel):
    stock: str
    price: float
    rsi: float
    ma_25: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    volume: float
    timestamp: str
    signal_triggered: Optional[bool] = False
    signal_type: Optional[str] = None

# ==================== Global State ====================

class TradingState:
    def __init__(self):
        self.config: Optional[MonitorConfig] = None
        self.signals: Dict[str, SignalLog] = {}
        self.indicators: Dict[str, IndicatorData] = {}
        self.monitoring = False
        
    def add_signal(self, stock: str, entry_price: float, investment: float):
        signal_id = f"{stock}_{datetime.now().timestamp()}"
        self.signals[signal_id] = SignalLog(
            stock=stock,
            signal_type="BUY",
            entry_price=entry_price,
            entry_time=datetime.now().isoformat(),
            investment=investment,
            status="ACTIVE"
        )
        logger.info(f"Signal added: {stock} @ ₹{entry_price}")
        return signal_id
    
    def update_signal_pnl(self, signal_id: str, current_price: float):
        if signal_id in self.signals:
            signal = self.signals[signal_id]
            signal.current_price = current_price
            signal.current_pnl = (current_price - signal.entry_price) * (signal.investment / signal.entry_price)
            signal.current_pnl_percent = ((current_price - signal.entry_price) / signal.entry_price) * 100

state = TradingState()

# ==================== Indicator Calculations ====================

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0
    
    prices = prices[-period-14:]  # Get enough data
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return prices[-1], prices[-1], prices[-1]
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    return float(upper), float(sma), float(lower)

def calculate_moving_average(prices: np.ndarray, period: int = 25) -> float:
    """Calculate Moving Average"""
    if len(prices) < period:
        return float(np.mean(prices))
    return float(np.mean(prices[-period:]))

def detect_kotegawa_signal(indicators: Dict) -> Optional[str]:
    """
    Detect Kotegawa trading signals:
    - RSI < 30 (oversold) + price below 25-MA = BUY signal
    - OR RSI < 35 + price at lower Bollinger Band
    """
    rsi = indicators.get('rsi', 50)
    price = indicators.get('price', 0)
    ma_25 = indicators.get('ma_25', 0)
    bb_lower = indicators.get('bb_lower', 0)
    
    # Main signal: RSI oversold + price below MA
    if rsi < 30 and price < ma_25 * 1.02:  # Small buffer for price
        return "BUY"
    
    # Alternative: Price at lower Bollinger Band with low RSI
    if rsi < 35 and price <= bb_lower * 1.01:
        return "BUY"
    
    return None

# ==================== Data Fetching with yfinance ====================

def get_stock_data(stock: str, days: int = 50) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance
    Converts NSE symbol to yfinance format (e.g., TCS -> TCS.NS)
    """
    try:
        # Add .NS suffix for NSE stocks if not already present
        yf_symbol = stock if stock.endswith('.NS') else f"{stock}.NS"
        
        # Fetch historical data
        data = yf.download(
            yf_symbol,
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            progress=False,
            interval="1d"
        )
        
        if data.empty:
            logger.warning(f"No data for {stock}")
            return None
        
        # Rename columns to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data for {stock}: {str(e)}")
        return None

# ==================== API Routes ====================

@app.post("/api/config")
async def set_config(config: MonitorConfig):
    """Set monitoring configuration"""
    try:
        state.config = config
        
        # Validate stocks can be fetched
        test_stocks = config.stocks[:2]  # Test first 2
        for stock in test_stocks:
            data = get_stock_data(stock, days=7)
            if data is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot fetch data for {stock}. Check symbol (e.g., TCS, RELIANCE, HDFC)"
                )
        
        logger.info(f"Configured {len(config.stocks)} stocks: {config.stocks}")
        
        return {
            "status": "success",
            "message": "Configuration set",
            "stocks": config.stocks,
            "investment_per_signal": config.investment_per_signal
        }
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/start-monitoring")
async def start_monitoring():
    """Start monitoring stocks for Kotegawa signals"""
    if not state.config:
        raise HTTPException(status_code=400, detail="Configuration not set")
    
    state.monitoring = True
    
    # Start background monitoring task
    asyncio.create_task(monitoring_loop())
    
    return {
        "status": "Monitoring started",
        "stocks": state.config.stocks,
        "check_interval": state.config.check_interval
    }

@app.post("/api/stop-monitoring")
async def stop_monitoring():
    """Stop monitoring stocks"""
    state.monitoring = False
    return {"status": "Monitoring stopped"}

@app.get("/api/signals")
async def get_signals():
    """Get all active signals with current P&L"""
    signals_list = []
    for signal_id, signal in state.signals.items():
        signal_dict = signal.dict()
        signal_dict["id"] = signal_id
        signals_list.append(signal_dict)
    
    # Calculate statistics
    total_signals = len(signals_list)
    winning_signals = len([s for s in signals_list if s.get('current_pnl_percent', 0) > 0])
    total_pnl = sum(s.get('current_pnl', 0) for s in signals_list)
    avg_pnl_percent = sum(s.get('current_pnl_percent', 0) for s in signals_list) / total_signals if total_signals > 0 else 0
    
    return {
        "signals": signals_list,
        "stats": {
            "total": total_signals,
            "winning": winning_signals,
            "win_rate": (winning_signals / total_signals * 100) if total_signals > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl_percent": avg_pnl_percent
        }
    }

@app.get("/api/indicators/{stock}")
async def get_indicators(stock: str):
    """Get latest indicators for a stock"""
    if stock not in state.indicators:
        raise HTTPException(status_code=404, detail=f"No data for {stock}")
    
    indicator = state.indicators[stock]
    return indicator.dict()

@app.get("/api/status")
async def get_status():
    """Get current monitoring status"""
    return {
        "monitoring": state.monitoring,
        "config_set": state.config is not None,
        "stocks": state.config.stocks if state.config else [],
        "total_signals": len(state.signals),
        "data_source": "Yahoo Finance (Free)",
        "update_time": datetime.now().isoformat()
    }

@app.get("/api/performance")
async def get_performance():
    """Get performance analytics"""
    signals_list = list(state.signals.values())
    
    if not signals_list:
        return {
            "total_signals": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "best_trade": 0,
            "worst_trade": 0
        }
    
    pnls = [s.current_pnl or 0 for s in signals_list]
    winning = sum(1 for p in pnls if p > 0)
    losing = sum(1 for p in pnls if p < 0)
    
    return {
        "total_signals": len(signals_list),
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": winning / len(signals_list) * 100 if signals_list else 0,
        "total_pnl": sum(pnls),
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        "best_trade": max(pnls) if pnls else 0,
        "worst_trade": min(pnls) if pnls else 0
    }

@app.post("/api/manual-signal")
async def manual_signal(stock: str, price: float):
    """Manually trigger a signal (for testing)"""
    signal_id = state.add_signal(stock, price, state.config.investment_per_signal if state.config else 10000)
    return {"signal_id": signal_id, "status": "created"}

@app.delete("/api/signals/{signal_id}")
async def close_signal(signal_id: str):
    """Close a signal (mark as completed)"""
    if signal_id not in state.signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    state.signals[signal_id].status = "CLOSED"
    return {"status": "Signal closed"}

@app.get("/")
async def root():
    return {
        "message": "Kotegawa Trading Monitor - Free Version",
        "data_source": "Yahoo Finance (yfinance)",
        "status": "running",
        "version": "2.0-free"
    }

# ==================== Background Monitoring Loop ====================

async def monitoring_loop():
    """Main monitoring loop - runs in background"""
    while state.monitoring:
        try:
            if not state.config:
                await asyncio.sleep(60)
                continue
            
            logger.info(f"Fetching data for {len(state.config.stocks)} stocks...")
            
            # Fetch data for each stock
            for stock in state.config.stocks:
                try:
                    # Get historical data
                    data = get_stock_data(stock, days=50)
                    
                    if data is None or data.empty:
                        logger.warning(f"Skipping {stock} - no data")
                        continue
                    
                    # Extract OHLCV
                    prices = data['close'].values
                    volumes = data['volume'].values
                    
                    if len(prices) < 25:  # Need at least 25 days for MA
                        continue
                    
                    # Calculate indicators
                    rsi = calculate_rsi(prices)
                    ma_25 = calculate_moving_average(prices)
                    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
                    current_price = float(prices[-1])
                    current_volume = float(volumes[-1])
                    
                    # Check for signals
                    signal = detect_kotegawa_signal({
                        'rsi': rsi,
                        'price': current_price,
                        'ma_25': ma_25,
                        'bb_lower': bb_lower
                    })
                    
                    # Store indicator data
                    state.indicators[stock] = IndicatorData(
                        stock=stock,
                        price=current_price,
                        rsi=rsi,
                        ma_25=ma_25,
                        bb_upper=bb_upper,
                        bb_lower=bb_lower,
                        bb_middle=bb_middle,
                        volume=current_volume,
                        timestamp=datetime.now().isoformat(),
                        signal_triggered=signal is not None,
                        signal_type=signal
                    )
                    
                    # Add new signal if triggered
                    if signal == "BUY":
                        # Check if this stock already has an active signal
                        has_active = any(
                            s.stock == stock and s.status == "ACTIVE"
                            for s in state.signals.values()
                        )
                        if not has_active:
                            logger.info(f"🎯 Signal for {stock} @ ₹{current_price:.2f} (RSI: {rsi:.1f})")
                            state.add_signal(stock, current_price, state.config.investment_per_signal)
                    
                    # Update P&L for active signals
                    for signal_id, signal_obj in state.signals.items():
                        if signal_obj.stock == stock and signal_obj.status == "ACTIVE":
                            state.update_signal_pnl(signal_id, current_price)
                
                except Exception as e:
                    logger.error(f"Error processing {stock}: {str(e)}")
                    continue
            
            logger.info(f"Update complete. Active signals: {len([s for s in state.signals.values() if s.status == 'ACTIVE'])}")
            
            # Wait for next check
            await asyncio.sleep(state.config.check_interval if state.config else 3600)
        
        except Exception as e:
            logger.error(f"Monitoring loop error: {str(e)}")
            await asyncio.sleep(60)

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check for deployment monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "monitoring": state.monitoring,
        "signals_count": len(state.signals)
    }

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for Railway/cloud deployment
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
