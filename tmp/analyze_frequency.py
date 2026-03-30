import re
from datetime import datetime

log_path = r'c:\Users\Juampa\Desktop\tradingBot\logs\trading_bot.log'
pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .* Cycle #\d+ complete .* Signal: (\w+) \((\d+\.\d+)\) \| AI: \w+ \((\d+\.\d+)\)')

candidates = []

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            timestamp_str, action, strength, ai_conf = match.groups()
            strength = float(strength)
            ai_conf = float(ai_conf)
            candidates.append({
                'timestamp': timestamp_str,
                'action': action,
                'strength': strength,
                'ai_conf': ai_conf,
                'line': line.strip()
            })

# 1. Total opportunities missed due to 0.5 threshold
missed_threshold = [c for c in candidates if 0.2 <= c['strength'] < 0.5]
print(f"SEÑALES FILTRADAS POR UMBRAL DE FUERZA (0.2 - 0.5): {len(missed_threshold)}")

# 2. Total opportunities missed due to AI Confidence (assuming strength was > 0.5)
missed_ai = [c for c in candidates if c['strength'] >= 0.5 and c['ai_conf'] < 0.65]
print(f"SEÑALES FILTRADAS POR BAJA CONFIANZA IA (< 0.65): {len(missed_ai)}")

# 3. Total actual execution attempts (strength >= 0.5 and AI potentially adjusted or not)
# Let's count how many times "TRADE EXECUTED" shows up
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    exec_count = sum(1 for line in f if "TRADE EXECUTED" in line)
print(f"TOTAL EJECUCIONES REALES: {exec_count}")

# 4. Show the last 5 missed due to 0.5 strength
print("\nÚLTIMAS 5 SEÑALES QUE NO LLEGARON AL 0.5 DE FUERZA:")
for c in missed_threshold[-5:]:
    print(f"[{c['timestamp']}] {c['action']} (Fuerza: {c['strength']:.2f})")
