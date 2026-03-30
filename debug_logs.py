import os
from pathlib import Path
from config import settings

log_file = settings.BASE_DIR / "logs" / "trading_bot.log"
print(f"Log file path: {log_file}")
print(f"Exists: {log_file.exists()}")
if log_file.exists():
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"Number of lines: {len(lines)}")
            if lines:
                print(f"Last line: {lines[-1].strip()}")
            else:
                print("File is empty or could not read lines")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File does not exist")
