# Configuração Automática de Chave SSH
# Este script configura autenticação sem senha

$SERVER = "right-scotia-humor-toward.trycloudflare.com"
$USER = "root"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Configuração de Chave SSH" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Verificar se já existe chave SSH
$sshKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
$sshPubKeyPath = "$sshKeyPath.pub"

if (-Not (Test-Path $sshKeyPath)) {
    Write-Host "[1/3] Gerando chave SSH..." -ForegroundColor Yellow

    # Criar diretório .ssh se não existir
    $sshDir = "$env:USERPROFILE\.ssh"
    if (-Not (Test-Path $sshDir)) {
        New-Item -ItemType Directory -Path $sshDir | Out-Null
    }

    # Gerar chave SSH
    & ssh-keygen -t ed25519 -f $sshKeyPath -N '""' -C "DSL-AG-upload"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK - Chave gerada em: $sshKeyPath`n" -ForegroundColor Green
    } else {
        Write-Host "ERRO ao gerar chave`n" -ForegroundColor Red
        pause
        exit 1
    }
} else {
    Write-Host "[1/3] Chave SSH já existe" -ForegroundColor Green
    Write-Host "     Localização: $sshKeyPath`n" -ForegroundColor Gray
}

# Copiar chave pública para o servidor
Write-Host "[2/3] Copiando chave pública para o servidor..." -ForegroundColor Yellow
Write-Host "     Servidor: $USER@$SERVER" -ForegroundColor Gray
Write-Host "     IMPORTANTE: Você precisará digitar a SENHA DO SERVIDOR`n" -ForegroundColor Yellow

# Ler chave pública
$pubKey = Get-Content $sshPubKeyPath

# Comando para adicionar chave no servidor
$addKeyCmd = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pubKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo OK"

Write-Host "     Digite a senha quando solicitado:" -ForegroundColor Gray
$result = & ssh -o StrictHostKeyChecking=no "${USER}@${SERVER}" $addKeyCmd 2>&1

if ($result -match "OK") {
    Write-Host "     OK - Chave copiada com sucesso!`n" -ForegroundColor Green
} else {
    Write-Host "     ERRO ao copiar chave" -ForegroundColor Red
    Write-Host "     Resposta: $result`n" -ForegroundColor Yellow
    pause
    exit 1
}

# Testar conexão sem senha
Write-Host "[3/3] Testando conexão sem senha..." -ForegroundColor Yellow
$testResult = & ssh -o StrictHostKeyChecking=no "${USER}@${SERVER}" "echo TESTE_OK" 2>&1

if ($testResult -match "TESTE_OK") {
    Write-Host "     OK - Autenticação sem senha funcionando!`n" -ForegroundColor Green
} else {
    Write-Host "     AVISO: Ainda pedindo senha" -ForegroundColor Yellow
    Write-Host "     Resposta: $testResult`n" -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CONFIGURAÇÃO CONCLUÍDA!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Agora você pode usar o upload.bat sem digitar senha!`n" -ForegroundColor White

pause
