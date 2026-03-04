# Upload com Senha - Versão Final Simplificada
# Uso: Envia arquivos via SCP com senha

param(
    [string]$Server = "right-scotia-humor-toward.trycloudflare.com",
    [string]$User = "root",
    [string]$RemotePath = "~/DSL-AG-hybrid"
)

$LOCAL_BASE = $PSScriptRoot
$FILES = @("ga_operators.py", "ga.py", "main.py", "config.yaml")

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Upload SSH - DSL-AG-hybrid" -ForegroundColor Cyan
Write-Host "Servidor: $User@$Server" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "NOTA: Você precisará digitar a senha para cada arquivo" -ForegroundColor Yellow
Write-Host "      (É normal no Windows sem chave SSH configurada)`n" -ForegroundColor Gray

# Criar diretório remoto primeiro
Write-Host "[1/2] Criando diretório remoto..." -ForegroundColor Yellow
Write-Host "Digite a senha:" -ForegroundColor Gray
& ssh -o StrictHostKeyChecking=no "${User}@${Server}" "mkdir -p $RemotePath" 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Diretório pronto`n" -ForegroundColor Green
} else {
    Write-Host "AVISO - Prosseguindo mesmo assim`n" -ForegroundColor Yellow
}

# Upload dos arquivos
Write-Host "[2/2] Enviando arquivos...`n" -ForegroundColor Yellow

$success = 0
$failed = 0
$fileNumber = 1

foreach ($file in $FILES) {
    $localFile = Join-Path $LOCAL_BASE $file

    if (-Not (Test-Path $localFile)) {
        Write-Host "  [$fileNumber/$($FILES.Count)] $file - NAO ENCONTRADO" -ForegroundColor Red
        $failed++
        $fileNumber++
        continue
    }

    $size = [math]::Round((Get-Item $localFile).Length / 1KB, 1)
    Write-Host "  [$fileNumber/$($FILES.Count)] Enviando: $file ($size KB)" -ForegroundColor White
    Write-Host "      Digite a senha:" -ForegroundColor Gray

    & scp -o StrictHostKeyChecking=no $localFile "${User}@${Server}:${RemotePath}/"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "      ✓ Enviado com sucesso`n" -ForegroundColor Green
        $success++
    } else {
        Write-Host "      ✗ Falhou`n" -ForegroundColor Red
        $failed++
    }

    $fileNumber++
}

Write-Host "========================================" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "✓ SUCESSO: $success/$($FILES.Count) arquivos enviados" -ForegroundColor Green
} else {
    Write-Host "⚠ PARCIAL: $success enviados, $failed falharam" -ForegroundColor Yellow
}
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Configuração do Sistema:" -ForegroundColor White
Write-Host "  • Hill Climbing Inteligente: ATIVADO" -ForegroundColor Green
Write-Host "  • Crossover Balanceado: ATIVADO" -ForegroundColor Green
Write-Host "`nPróximos passos:" -ForegroundColor White
Write-Host "  1. Conecte ao servidor: ssh $User@$Server" -ForegroundColor Gray
Write-Host "  2. Execute: cd $RemotePath" -ForegroundColor Gray
Write-Host "  3. Rode o experimento`n" -ForegroundColor Gray

pause
