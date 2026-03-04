param(
    [Parameter(Mandatory=$true)]
    [string]$SSHHost
)

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Blue
Write-Host "  TESTE SIMPLES: Workflow SSH + Google Drive" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Blue
Write-Host ""
Write-Host "SSH Host: $SSHHost" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

# Teste 1: Arquivos locais
Write-Host "CATEGORIA 1: Arquivos Locais" -ForegroundColor Yellow
Write-Host ""

Write-Host "  sync_to_colab.sh............ " -NoNewline
if (Test-Path "sync_to_colab.sh") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  setup_colab_remote.py....... " -NoNewline
if (Test-Path "setup_colab_remote.py") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  colab_workflow.ps1.......... " -NoNewline
if (Test-Path "colab_workflow.ps1") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  main.py..................... " -NoNewline
if (Test-Path "main.py") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  config.yaml................. " -NoNewline
if (Test-Path "config.yaml") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host ""

# Teste 2: SSH
Write-Host "CATEGORIA 2: Conectividade SSH" -ForegroundColor Yellow
Write-Host ""

Write-Host "  SSH acessivel............... " -NoNewline
$result = ssh $SSHHost "echo OK" 2>$null
if ($result -eq "OK") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  Pode criar diretorio........ " -NoNewline
ssh $SSHHost "mkdir -p /tmp/test && echo OK" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  Python disponivel........... " -NoNewline
$result = ssh $SSHHost "python3 --version" 2>$null
if ($result -match "Python 3") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host ""

# Teste 3: Google Colab
Write-Host "CATEGORIA 3: Google Colab Environment" -ForegroundColor Yellow
Write-Host ""

Write-Host "  Esta no Colab............... " -NoNewline
$result = ssh $SSHHost "test -d /content && echo OK" 2>$null
if ($result -match "OK") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  Google Drive montado........ " -NoNewline
$result = ssh $SSHHost "test -d /content/drive/MyDrive && echo OK" 2>$null
if ($result -match "OK") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  Pode escrever no Drive...... " -NoNewline
$testFile = "/content/drive/MyDrive/.test_$(Get-Date -Format 'yyyyMMddHHmmss').txt"
ssh $SSHHost "echo test > $testFile && test -f $testFile && rm $testFile && echo OK" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host ""

# Teste 4: Dependencias Python
Write-Host "CATEGORIA 4: Dependencias Python" -ForegroundColor Yellow
Write-Host ""

Write-Host "  google.colab................ " -NoNewline
$result = ssh $SSHHost "python3 -c 'import google.colab; print(`"OK`")'" 2>$null
if ($result -match "OK") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host "  yaml........................ " -NoNewline
$result = ssh $SSHHost "python3 -c 'import yaml; print(`"OK`")'" 2>$null
if ($result -match "OK") { Write-Host "OK" -ForegroundColor Green; $passed++ } else { Write-Host "FAIL" -ForegroundColor Red; $failed++ }

Write-Host ""

# Resumo
Write-Host "=========================================================================" -ForegroundColor Blue
Write-Host "  RESUMO" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Blue
Write-Host ""

$total = $passed + $failed
Write-Host "Total de testes: $total"
Write-Host "Passou: $passed" -ForegroundColor Green
Write-Host "Falhou: $failed" -ForegroundColor Red
Write-Host ""

if ($failed -eq 0) {
    Write-Host "TODOS OS TESTES PASSARAM!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Proximo passo:" -ForegroundColor Cyan
    Write-Host "  .\colab_workflow.ps1 -SSHHost $SSHHost -Action all -Stream RBF_Abrupt_Severe -Chunks 1"
    Write-Host ""
    exit 0
} else {
    Write-Host "ALGUNS TESTES FALHARAM" -ForegroundColor Red
    Write-Host ""

    # Diagnostico
    $driveTest = ssh $SSHHost "test -d /content/drive/MyDrive && echo OK" 2>$null
    if ($driveTest -ne "OK") {
        Write-Host "PROBLEMA: Google Drive NAO esta montado" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "SOLUCAO:" -ForegroundColor Cyan
        Write-Host "  1. Abra o notebook Colab no navegador"
        Write-Host "  2. Execute:"
        Write-Host "     from google.colab import drive"
        Write-Host "     drive.mount('/content/drive')"
        Write-Host ""
    }

    exit 1
}
