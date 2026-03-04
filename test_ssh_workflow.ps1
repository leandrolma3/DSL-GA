#Requires -Version 5.1
<#
.SYNOPSIS
    Script de teste para validar o workflow SSH + Google Drive

.DESCRIPTION
    Executa uma série de testes para garantir que o workflow está configurado corretamente

.PARAMETER SSHHost
    Host SSH do Cloudflare Tunnel

.EXAMPLE
    .\test_ssh_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com

.NOTES
    Autor: Claude Code
    Data: 2025-10-18
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$SSHHost
)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

function Write-TestHeader {
    param([string]$Text)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
}

function Test-Step {
    param(
        [string]$Description,
        [ScriptBlock]$Test,
        [ref]$PassedCount,
        [ref]$FailedCount
    )

    Write-Host "🔹 $Description... " -NoNewline

    try {
        $result = & $Test
        if ($result) {
            Write-Host "✅ PASS" -ForegroundColor Green
            $PassedCount.Value++
            return $true
        } else {
            Write-Host "❌ FAIL" -ForegroundColor Red
            $FailedCount.Value++
            return $false
        }
    } catch {
        Write-Host "❌ ERROR: $_" -ForegroundColor Red
        $FailedCount.Value++
        return $false
    }
}

# ============================================================================
# TESTES
# ============================================================================

$passed = 0
$failed = 0

Write-TestHeader "TESTE DE VALIDAÇÃO: Workflow SSH + Google Drive"

Write-Host "🔧 SSH Host: $SSHHost" -ForegroundColor Cyan
Write-Host ""

# Teste 1: Arquivos locais existem
Write-TestHeader "CATEGORIA 1: Arquivos Locais"

Test-Step "Arquivo sync_to_colab.sh existe" {
    Test-Path "sync_to_colab.sh"
} ([ref]$passed) ([ref]$failed)

Test-Step "Arquivo setup_colab_remote.py existe" {
    Test-Path "setup_colab_remote.py"
} ([ref]$passed) ([ref]$failed)

Test-Step "Arquivo colab_workflow.ps1 existe" {
    Test-Path "colab_workflow.ps1"
} ([ref]$passed) ([ref]$failed)

Test-Step "Arquivo main.py existe" {
    Test-Path "main.py"
} ([ref]$passed) ([ref]$failed)

Test-Step "Arquivo config.yaml existe" {
    Test-Path "config.yaml"
} ([ref]$passed) ([ref]$failed)

# Teste 2: Conectividade SSH
Write-TestHeader "CATEGORIA 2: Conectividade SSH"

Test-Step "SSH está acessível" {
    $result = ssh $SSHHost "echo OK" 2>$null
    $result -eq "OK"
} ([ref]$passed) ([ref]$failed)

Test-Step "Pode criar diretório remoto" {
    ssh $SSHHost "mkdir -p /tmp/test_workflow && echo OK" 2>$null | Out-Null
    $LASTEXITCODE -eq 0
} ([ref]$passed) ([ref]$failed)

Test-Step "Pode executar comandos Python" {
    $result = ssh $SSHHost "python3 --version" 2>$null
    $result -match "Python 3"
} ([ref]$passed) ([ref]$failed)

# Teste 3: Google Colab environment
Write-TestHeader "CATEGORIA 3: Google Colab Environment"

Test-Step "Está no Google Colab" {
    $result = ssh $SSHHost "python3 -c 'import sys; print(`"/content`" in sys.path)'" 2>$null
    $result -match "True"
} ([ref]$passed) ([ref]$failed)

Test-Step "Google Drive está montado" {
    ssh $SSHHost "test -d /content/drive/MyDrive && echo OK" 2>$null | Out-Null
    $LASTEXITCODE -eq 0
} ([ref]$passed) ([ref]$failed)

Test-Step "Pode escrever no Drive" {
    $testFile = "/content/drive/MyDrive/.test_workflow_$(Get-Date -Format 'yyyyMMddHHmmss').txt"
    ssh $SSHHost "echo 'test' > $testFile && test -f $testFile && rm $testFile && echo OK" 2>$null | Out-Null
    $LASTEXITCODE -eq 0
} ([ref]$passed) ([ref]$failed)

# Teste 4: Upload de arquivos (teste com arquivo pequeno)
Write-TestHeader "CATEGORIA 4: Upload de Arquivos (SCP)"

Test-Step "SCP está funcional" {
    $testContent = "test_$(Get-Date -Format 'yyyyMMddHHmmss')"
    $testFile = "test_upload.txt"
    $testContent | Out-File -FilePath $testFile -Encoding utf8

    scp -q $testFile "${SSHHost}:/tmp/" 2>$null
    $scpResult = $LASTEXITCODE -eq 0

    Remove-Item $testFile -ErrorAction SilentlyContinue
    ssh $SSHHost "rm /tmp/$testFile" 2>$null

    $scpResult
} ([ref]$passed) ([ref]$failed)

# Teste 5: Dependências Python
Write-TestHeader "CATEGORIA 5: Dependências Python"

Test-Step "Módulo google.colab disponível" {
    $result = ssh $SSHHost "python3 -c 'import google.colab; print(`"OK`")'" 2>$null
    $result -match "OK"
} ([ref]$passed) ([ref]$failed)

Test-Step "Módulo yaml disponível" {
    $result = ssh $SSHHost "python3 -c 'import yaml; print(`"OK`")'" 2>$null
    $result -match "OK"
} ([ref]$passed) ([ref]$failed)

# Resumo
Write-TestHeader "RESUMO DOS TESTES"

$total = $passed + $failed

Write-Host ""
Write-Host "Total de testes: $total" -ForegroundColor Cyan
Write-Host "✅ Passou: $passed" -ForegroundColor Green
Write-Host "❌ Falhou: $failed" -ForegroundColor Red
Write-Host ""

if ($failed -eq 0) {
    Write-Host "╔══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                  🎉 TODOS OS TESTES PASSARAM! 🎉                    ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ Seu ambiente está configurado corretamente!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Próximos passos:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Executar workflow completo:" -ForegroundColor Yellow
    Write-Host "   .\colab_workflow.ps1 -SSHHost $SSHHost -Action all -Stream RBF_Abrupt_Severe -Chunks 1"
    Write-Host ""
    Write-Host "2. Ou executar passo a passo:" -ForegroundColor Yellow
    Write-Host "   .\colab_workflow.ps1 -SSHHost $SSHHost -Action sync"
    Write-Host "   .\colab_workflow.ps1 -SSHHost $SSHHost -Action setup"
    Write-Host "   .\colab_workflow.ps1 -SSHHost $SSHHost -Action run -Stream RBF_Abrupt_Severe -Chunks 1"
    Write-Host ""

    exit 0
} else {
    Write-Host "╔══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║                  ⚠️  ALGUNS TESTES FALHARAM ⚠️                      ║" -ForegroundColor Red
    Write-Host "╚══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    Write-Host "⚠️  Problemas detectados:" -ForegroundColor Yellow
    Write-Host ""

    # Diagnóstico
    if ((ssh $SSHHost "test -d /content/drive/MyDrive && echo OK" 2>$null) -ne "OK") {
        Write-Host "❌ Google Drive NÃO está montado" -ForegroundColor Red
        Write-Host ""
        Write-Host "SOLUÇÃO:" -ForegroundColor Yellow
        Write-Host "  1. Abra o notebook Colab no navegador"
        Write-Host "  2. Execute esta célula:"
        Write-Host ""
        Write-Host "     from google.colab import drive" -ForegroundColor Cyan
        Write-Host "     drive.mount('/content/drive')" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  3. Autorize quando solicitado"
        Write-Host "  4. Execute este teste novamente"
        Write-Host ""
    }

    if ((ssh $SSHHost "echo OK" 2>$null) -ne "OK") {
        Write-Host "❌ SSH não está acessível" -ForegroundColor Red
        Write-Host ""
        Write-Host "SOLUÇÃO:" -ForegroundColor Yellow
        Write-Host "  1. Verifique se o notebook colab_ssh_setup.ipynb está rodando"
        Write-Host "  2. Verifique se o host está correto: $SSHHost"
        Write-Host "  3. Tente conectar manualmente: ssh $SSHHost"
        Write-Host ""
    }

    exit 1
}
