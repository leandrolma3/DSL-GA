# Upload Simples para Servidor SSH
# Versão: 2.0 - Simplificada

# Configuração
$CONFIG_FILE = "server_address.txt"
if (Test-Path $CONFIG_FILE) {
    $SERVER = (Get-Content $CONFIG_FILE).Trim()
} else {
    $SERVER = "ssh right-scotia-humor-toward.trycloudflare.com"
}

$REMOTE_PATH = "~/DSL-AG-hybrid"
$LOCAL_BASE = $PSScriptRoot

$FILES = @("ga_operators.py", "ga.py", "main.py", "config.yaml")

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Upload para: $SERVER" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Teste de conexão
Write-Host "[1/3] Testando conexão..." -ForegroundColor Yellow
$testResult = & ssh $SERVER 'echo OK' 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO: Falha na conexão SSH" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "OK - Conectado`n" -ForegroundColor Green

# Criar diretório remoto
Write-Host "[2/3] Verificando diretório remoto..." -ForegroundColor Yellow
& ssh $SERVER ('mkdir -p ' + $REMOTE_PATH) 2>&1 | Out-Null
Write-Host "OK - Diretório pronto`n" -ForegroundColor Green

# Upload dos arquivos
Write-Host "[3/3] Enviando arquivos..." -ForegroundColor Yellow
$success = 0
$failed = 0

foreach ($file in $FILES) {
    $localFile = Join-Path $LOCAL_BASE $file
    if (Test-Path $localFile) {
        $size = [math]::Round((Get-Item $localFile).Length / 1KB, 1)
        Write-Host "  -> $file ($size KB)..." -NoNewline

        $dest = $SERVER + ':' + $REMOTE_PATH + '/'
        & scp $localFile $dest 2>&1 | Out-Null

        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            $success++
        } else {
            Write-Host " FALHOU" -ForegroundColor Red
            $failed++
        }
    } else {
        Write-Host "  -> $file... NAO ENCONTRADO" -ForegroundColor Yellow
        $failed++
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "SUCESSO: $success/$($FILES.Count) arquivos enviados" -ForegroundColor Green
} else {
    Write-Host "AVISO: $success sucesso, $failed falhas" -ForegroundColor Yellow
}
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Configuracao:"
Write-Host "  - HC Inteligente: ATIVADO" -ForegroundColor Green
Write-Host "  - Crossover Balanceado: ATIVADO`n" -ForegroundColor Green

pause
