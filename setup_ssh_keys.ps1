# Script para configurar autenticacao SSH sem senha
# Autor: Claude Code

param(
    [Parameter(Mandatory=$true)]
    [string]$SSHHost
)

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Blue
Write-Host "  SETUP: Autenticacao SSH sem Senha" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Blue
Write-Host ""

# Verifica se ja existe chave SSH
$sshDir = "$env:USERPROFILE\.ssh"
$keyFile = "$sshDir\id_rsa"

if (-not (Test-Path $keyFile)) {
    Write-Host "Gerando par de chaves SSH..." -ForegroundColor Yellow
    ssh-keygen -t rsa -b 4096 -f $keyFile -N '""' -q
    Write-Host "Chave SSH gerada: $keyFile" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "Chave SSH ja existe: $keyFile" -ForegroundColor Green
    Write-Host ""
}

# Le a chave publica
$pubKey = Get-Content "$keyFile.pub"

Write-Host "Configurando autenticacao no servidor..." -ForegroundColor Yellow
Write-Host ""
Write-Host "INSTRUCOES:" -ForegroundColor Cyan
Write-Host "============" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Conecte via SSH manualmente:" -ForegroundColor White
Write-Host "   ssh $SSHHost" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Execute estes comandos no servidor:" -ForegroundColor White
Write-Host ""
Write-Host "   mkdir -p ~/.ssh" -ForegroundColor Yellow
Write-Host "   chmod 700 ~/.ssh" -ForegroundColor Yellow
Write-Host "   echo '$pubKey' >> ~/.ssh/authorized_keys" -ForegroundColor Yellow
Write-Host "   chmod 600 ~/.ssh/authorized_keys" -ForegroundColor Yellow
Write-Host "   exit" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Teste a conexao (nao deve pedir senha):" -ForegroundColor White
Write-Host "   ssh $SSHHost 'echo OK'" -ForegroundColor Yellow
Write-Host ""

# Copia chave publica para clipboard se possivel
try {
    $pubKey | Set-Clipboard
    Write-Host "Chave publica copiada para clipboard!" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "Chave publica:" -ForegroundColor Cyan
    Write-Host $pubKey -ForegroundColor White
    Write-Host ""
}
