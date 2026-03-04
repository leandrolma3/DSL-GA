# Upload Simplificado - Sem travamentos
# Versao ultra-simples que funciona

$Server = "frozen-about-ball-indicating.trycloudflare.com"
$User = "root"
$RemotePath = "/content/DSL-AG-hybrid"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Upload Simples - DSL-AG-hybrid" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Servidor: $User@$Server"
Write-Host "Destino: $RemotePath"
Write-Host ""
Write-Host "NOTA: Voce digitara a senha 'root' DUAS vezes:"
Write-Host "  1x para criar diretorio"
Write-Host "  1x para upload em lote"
Write-Host ""
Write-Host "Pressione Enter para continuar..."
Read-Host

# Passo 1: Criar diretorio (UMA senha)
Write-Host ""
Write-Host "[1/2] Criando diretorio remoto..." -ForegroundColor Yellow
Write-Host "Digite a senha quando solicitado:" -ForegroundColor Green
Write-Host ""

ssh "$User@$Server" "mkdir -p $RemotePath"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERRO ao criar diretorio (codigo: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "OK - Diretorio criado" -ForegroundColor Green
Write-Host ""

# Passo 2: Upload em lote (UMA senha)
Write-Host "[2/2] Enviando TODOS os arquivos..." -ForegroundColor Yellow
Write-Host "Digite a senha quando solicitado:" -ForegroundColor Green
Write-Host ""

$files = @(
    "ga_operators.py",
    "ga.py",
    "main.py",
    "config.yaml",
    "setup_colab_remote.py",
    "compare_gbml_vs_river.py",
    "hill_climbing_v2.py",
    "intelligent_hc_strategies.py",
    "dt_rule_extraction.py",
    "early_stopping.py",
    "utils.py"
)

# Filtra apenas arquivos que existem
$existingFiles = @()
foreach ($f in $files) {
    if (Test-Path $f) {
        $existingFiles += $f
    } else {
        Write-Host "  Arquivo nao encontrado: $f" -ForegroundColor Yellow
    }
}

Write-Host "Enviando $($existingFiles.Count) arquivos..."
Write-Host ""

# Upload em lote (UMA unica senha)
scp $existingFiles "${User}@${Server}:${RemotePath}/"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCESSO!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "$($existingFiles.Count) arquivos enviados para $RemotePath" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERRO no upload (codigo: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    pause
    exit 1
}

# Instrucoes finais
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PROXIMOS PASSOS:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. MONTE O GOOGLE DRIVE (se ainda nao fez):" -ForegroundColor White
Write-Host "   No notebook Colab:" -ForegroundColor Yellow
Write-Host "     from google.colab import drive" -ForegroundColor Gray
Write-Host "     drive.mount('/content/drive')" -ForegroundColor Gray
Write-Host ""
Write-Host "2. CONECTE VIA SSH:" -ForegroundColor White
Write-Host "   ssh $User@$Server" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. DENTRO DO SSH, execute:" -ForegroundColor White
Write-Host "   cd $RemotePath" -ForegroundColor Yellow
Write-Host "   python3 setup_colab_remote.py" -ForegroundColor Yellow
Write-Host "   /root/run_experiment.sh python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 1" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

pause
