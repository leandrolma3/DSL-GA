import subprocess
import time
from datetime import datetime

# Nome do log
log_filename = f"experimento_teste_6chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
print(f"Log sera salvo em: {log_filename}")
print("Iniciando experimento...")
print("Tempo estimado: ~8 horas")
print("")

start_time = time.time()

# Executar main.py com config_test_single.yaml
# Verificar primeiro se main.py aceita argumento de config
process = subprocess.Popen(
    ['python', 'main.py', '--config', 'config_test_single.yaml'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Salvar output em arquivo e exibir em tempo real
with open(log_filename, 'w') as log_file:
    for line in process.stdout:
        print(line, end='')  # Exibe no Colab
        log_file.write(line)  # Salva no arquivo
        log_file.flush()  # Força escrita imediata

process.wait()
elapsed = time.time() - start_time

print("")
print("="*70)
print(f"Experimento concluido!")
print(f"Tempo total: {elapsed/3600:.2f} horas ({elapsed/60:.1f} minutos)")
print(f"Log salvo em: {log_filename}")
print("="*70)
