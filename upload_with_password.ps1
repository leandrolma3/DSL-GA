# Upload com Autenticação
# Versão: 2.1

# Configuração
$CONFIG_FILE = "server_address.txt"
if (Test-Path $CONFIG_FILE) {
    $SERVER = (Get-Content $CONFIG_FILE).Trim()
} else {
    $SERVER = "right-scotia-humor-toward.trycloudflare.com"
}

$USER = "root"
$REMOTE_PATH = "~/DSL-AG-hybrid"
$LOCAL_BASE = $PSScriptRoot

$FILES = @("ga_operators.py", "ga.py", "main.py", "config.yaml")

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Upload para: $USER@$SERVER" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Pedir senha uma vez
$password = Read-Host -AsSecureString "Digite a senha do servidor"
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($password)
$plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Teste de conexão
Write-Host "[1/3] Testando conexão..." -ForegroundColor Yellow
$testCmd = "echo '$plainPassword' | ssh -o StrictHostKeyChecking=no -T $USER@$SERVER 'echo OK' 2>&1"
$testResult = Invoke-Expression $testCmd

if ($testResult -notmatch "OK") {
    Write-Host "ERRO: Falha na conexão SSH" -ForegroundColor Red
    Write-Host "Verifique usuário/senha" -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host "OK - Conectado`n" -ForegroundColor Green

# Criar diretório remoto
Write-Host "[2/3] Criando diretório remoto..." -ForegroundColor Yellow
$mkdirCmd = "echo '$plainPassword' | ssh -o StrictHostKeyChecking=no -T $USER@$SERVER 'mkdir -p $REMOTE_PATH' 2>&1"
Invoke-Expression $mkdirCmd | Out-Null
Write-Host "OK - Diretório pronto`n" -ForegroundColor Green

# Upload dos arquivos usando pscp (ou scp com senha)
Write-Host "[3/3] Enviando arquivos..." -ForegroundColor Yellow
Write-Host "NOTA: Você precisará digitar a senha para cada arquivo`n" -ForegroundColor Yellow

$success = 0
$failed = 0

foreach ($file in $FILES) {
    $localFile = Join-Path $LOCAL_BASE $file
    if (Test-Path $localFile) {
        $size = [math]::Round((Get-Item $localFile).Length / 1KB, 1)
        Write-Host "  -> $file ($size KB)" -ForegroundColor White
        Write-Host "     Digite a senha quando solicitado:" -ForegroundColor Gray

        & scp -o StrictHostKeyChecking=no $localFile "${USER}@${SERVER}:${REMOTE_PATH}/"

        if ($LASTEXITCODE -eq 0) {
            Write-Host "     OK`n" -ForegroundColor Green
            $success++
        } else {
            Write-Host "     FALHOU`n" -ForegroundColor Red
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

# Limpar senha da memória
$plainPassword = $null

pause
