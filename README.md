# 🤖 AI Trading Bot Autónoma 24/7

Bot de trading impulsado por IA para operar en Binance (BTC/USDT) de forma autónoma. Combina estrategias técnicas clásicas con un modelo de Machine Learning (LSTM) para predicción de precios.

## 🚀 Características
- **Trading Autónomo 24/7**: Ejecución automática basada en ciclos programados.
- **Motor Híbrido**: Combina Tendencia, Reversión a la Media y Predicciones de IA.
- **IA Adaptativa**: Modelo LSTM que se re-entrena periódicamente.
- **Gestión de Riesgo Avanzada**: Position sizing, Stop-Loss/Take-Profit dinámico y límites de Drawdown.
- **Dashboard Web**: Visualización en tiempo real de precios, trades y balance.
- **Alertas Discord**: Notificaciones instantáneas de cada operación.
- **Modo Paper Trading**: Simulación realista con fees y slippage para pruebas sin riesgo.

## 🛠️ Instalación

1. **Clonar el repositorio** y entrar a la carpeta:
   ```bash
   cd tradingBot
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar el entorno**:
   - Copia `.env.example` a `.env`: `cp .env.example .env` (o `copy` en Windows).
   - Edita `.env` con tus API Keys de Binance y tu token de Discord.

## 🏃 Cómo Ejecutar

### 1. Entrenar la IA (Opcional pero recomendado)
Para que el bot tenga capacidad predictiva desde el inicio:
```bash
python main.py --train-only --epochs 50
```

### 2. Iniciar el Bot (Modo Paper Trading por defecto)
```bash
python main.py
```

### 3. Iniciar el Dashboard
En una terminal aparte:
```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
```
Luego abre `http://localhost:8080` en tu navegador.

## 📁 Estructura del Proyecto
- `ai/`: Modelos LSTM y procesamiento de features.
- `analysis/`: Indicadores técnicos (RSI, MACD, Bollinger, etc).
- `core/`: Orquestador principal y motor de ejecución.
- `data/`: Colección de datos y base de datos SQLite.
- `exchange/`: Conectores para Binance y Paper Trader.
- `risk/`: Gestión de riesgo y protección de capital.
- `strategies/`: Lógicas de trading (Trend, Mean Reversion).
- `dashboard/`: Dashboard web con FastAPI.
- `alerts/`: Sistema de notificaciones Discord.

## ⚠️ Descargo de Responsabilidad
Este software es para fines educativos. El trading de activos conlleva riesgos significativos de pérdida. Úsalo bajo tu propia responsabilidad.
