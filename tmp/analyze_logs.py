# Find signals that were "strong" but blocked by the 0.5 threshold
missed_signals = [c for c in candidates if 0.2 <= c['strength'] < 0.5]

print(f"SEÑALES 'CERCA' (Fuerza entre 0.2 y 0.5): {len(missed_signals)}")
if missed_signals:
    print("\nÚltimas 5 señales que casi se ejecutan:")
    for c in missed_signals[-5:]:
        print(f"{c['timestamp']} - Acción: {c['action']}, Fuerza: {c['strength']:.2f}")
        print(f"  Detalle: {c['line']}")

# Check AI Confidence blockage
ai_blocked = [c for c in candidates if c['strength'] >= 0.5 and c['ai_conf'] < 0.65]
print(f"\nSEÑALES BLOQUEADAS POR BAJA CONFIANZA DE IA (>0.5 fuerza pero <0.65 IA): {len(ai_blocked)}")
