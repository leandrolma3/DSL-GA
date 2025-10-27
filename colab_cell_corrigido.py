import os
from datetime import datetime

# Criar nome do arquivo com timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'/content/drive/MyDrive/DSL-AG-hybrid/experiments/drift_test_6chunks_{timestamp}.log'

# ✅ CORREÇÃO: Garante que o diretório existe antes de usar tee
log_dir = os.path.dirname(log_file)
os.makedirs(log_dir, exist_ok=True)

print(f"🚀 Iniciando experimento...")
print(f"📁 Log será salvo em: {log_file}")
print(f"⏱️ Tempo estimado: 12-15 horas\n")
print("="*60 + "\n")

# Executar com tee (mostra na tela E salva no arquivo)
!python3 main.py 2>&1 | tee "{log_file}"

print(f"\n✅ Execução finalizada!")
print(f"📄 Log completo salvo em: {log_file}")
