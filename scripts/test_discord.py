import asyncio
import sys
from pathlib import Path

# Añadir el directorio raíz al path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

from alerts.discord_bot import notifier

async def main():
    print("Iniciando prueba de Discord...")
    try:
        await notifier.send_message(
            content="✅ **¡Conexión Exitosa!**\nEl bot de trading ahora está vinculado correctamente a este canal y listo para enviar alertas.",
            title="Prueba de Sistema - Trading Bot",
            color=0x2ecc71
        )
        print("¡Mensaje enviado con éxito! Revisa tu canal de Discord.")
    except Exception as e:
        print(f"Error detectado durante la prueba: {e}")
    finally:
        await notifier.close()

if __name__ == "__main__":
    asyncio.run(main())
