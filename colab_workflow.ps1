#Requires -Version 5.1
<#
.SYNOPSIS
    Workflow completo para executar experimentos no Google Colab via SSH

.DESCRIPTION
    Este script automatiza:
    1. Upload de arquivos para Colab via SCP
    2. Setup do ambiente (Drive, logging, etc.)
    3. Execução de experimentos
    4. Download de resultados

.PARAMETER SSHHost
    Host SSH do Cloudflare Tunnel (ex: logged-minerals-axis-infrastructure.trycloudflare.com)

.PARAMETER Action
    Ação a executar: sync, setup, run, download, all

.PARAMETER Stream
    Nome do stream (ex: RBF_Abrupt_Severe)

.PARAMETER Chunks
    Número de chunks (padrão: 3)

.EXAMPLE
    # Workflow completo (sync + setup + run)
    .\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action all -Stream RBF_Abrupt_Severe -Chunks 3

.EXAMPLE
    # Apenas sync
    .\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action sync

.EXAMPLE
    # Apenas setup
    .\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action setup

.EXAMPLE
    # Executar experimento
    .\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action run -Stream RBF_Abrupt_Severe -Chunks 3

.NOTES
    Autor: Claude Code
    Data: 2025-10-18
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$SSHHost,

    [Parameter(Mandatory=$false)]
    [ValidateSet('sync', 'setup', 'run', 'download', 'all')]
    [string]$Action = 'all',

    [Parameter(Mandatory=$false)]
    [string]$Stream = 'RBF_Abrupt_Severe',

    [Parameter(Mandatory=$false)]
    [int]$Chunks = 3,

    [Parameter(Mandatory=$false)]
    [int]$Population = 120,

    [Parameter(Mandatory=$false)]
    [int]$MaxGenerations = 200
)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
}

function Write-Step {
    param([string]$Text)
    Write-Host "🔹 $Text" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Text)
    Write-Host "✅ $Text" -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host "❌ $Text" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Text)
    Write-Host "⚠️  $Text" -ForegroundColor Yellow
}

# ============================================================================
# ETAPA 1: SYNC (Upload de arquivos)
# ============================================================================

function Invoke-Sync {
    Write-Header "ETAPA 1: SYNC - Upload de Arquivos"

    $FilesToSync = @(
        "main.py",
        "ga.py",
        "ga_operators.py",
        "config.yaml",
        "compare_gbml_vs_river.py",
        "hill_climbing_v2.py",
        "intelligent_hc_strategies.py",
        "dt_rule_extraction.py",
        "colab_drive_setup.py",
        "setup_colab_remote.py",
        "early_stopping.py",
        "utils.py",
        "requirements.txt"
    )

    $RemoteDir = "/root/DSL-AG-hybrid"

    # Cria diretório remoto
    Write-Step "Criando diretório remoto..."
    $createDirCmd = "ssh $SSHHost `"mkdir -p $RemoteDir`""
    Invoke-Expression $createDirCmd | Out-Null

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Diretório criado: $RemoteDir"
    } else {
        Write-Error "Falha ao criar diretório remoto"
        return $false
    }

    # Upload de arquivos
    $successCount = 0
    $failCount = 0
    $total = $FilesToSync.Count

    Write-Step "Enviando $total arquivos..."
    Write-Host ""

    foreach ($file in $FilesToSync) {
        if (Test-Path $file) {
            Write-Host "  📤 $file... " -NoNewline

            $scpCmd = "scp -q `"$file`" ${SSHHost}:${RemoteDir}/"
            Invoke-Expression $scpCmd 2>$null

            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅" -ForegroundColor Green
                $successCount++
            } else {
                Write-Host "❌" -ForegroundColor Red
                $failCount++
            }
        } else {
            Write-Host "  ⚠️  $file - SKIP (não existe)" -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Success "Sync concluído: $successCount OK, $failCount falhas"

    return ($failCount -eq 0)
}

# ============================================================================
# ETAPA 2: SETUP (Configuração do ambiente)
# ============================================================================

function Invoke-Setup {
    Write-Header "ETAPA 2: SETUP - Configuração do Ambiente"

    Write-Step "Executando setup remoto..."

    $setupCmd = "ssh $SSHHost `"cd /root/DSL-AG-hybrid && python setup_colab_remote.py`""

    Write-Host ""
    Invoke-Expression $setupCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Success "Setup concluído com sucesso!"
        return $true
    } else {
        Write-Host ""
        Write-Error "Setup falhou"
        Write-Warning "Certifique-se de que o Google Drive está montado no Colab"
        return $false
    }
}

# ============================================================================
# ETAPA 3: RUN (Execução do experimento)
# ============================================================================

function Invoke-Run {
    Write-Header "ETAPA 3: RUN - Executando Experimento"

    Write-Host "📊 Parâmetros:" -ForegroundColor Cyan
    Write-Host "   Stream:         $Stream"
    Write-Host "   Chunks:         $Chunks"
    Write-Host "   População:      $Population"
    Write-Host "   Max Gerações:   $MaxGenerations"
    Write-Host ""

    Write-Step "Iniciando experimento..."
    Write-Host ""

    $runCmd = @"
ssh $SSHHost "cd /root/DSL-AG-hybrid && /root/run_experiment.sh python compare_gbml_vs_river.py --stream $Stream --chunks $Chunks --population $Population --max-generations $MaxGenerations"
"@

    Invoke-Expression $runCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Success "Experimento concluído!"
        return $true
    } else {
        Write-Host ""
        Write-Error "Experimento falhou (código: $LASTEXITCODE)"
        return $false
    }
}

# ============================================================================
# ETAPA 4: DOWNLOAD (Download de resultados)
# ============================================================================

function Invoke-Download {
    Write-Header "ETAPA 4: DOWNLOAD - Baixando Resultados"

    Write-Step "Procurando último experimento..."

    # Busca último experimento
    $findCmd = "ssh $SSHHost `"ls -td /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_* 2>/dev/null | head -1`""
    $latestExperiment = Invoke-Expression $findCmd

    if ([string]::IsNullOrWhiteSpace($latestExperiment)) {
        Write-Warning "Nenhum experimento encontrado no Drive"
        return $false
    }

    Write-Success "Encontrado: $latestExperiment"

    # Diretório local para download
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $localDir = ".\results_download_$timestamp"

    New-Item -ItemType Directory -Path $localDir -Force | Out-Null

    Write-Step "Baixando para: $localDir"

    # Download via SCP recursivo
    $scpCmd = "scp -r ${SSHHost}:${latestExperiment}/* `"$localDir\`""

    Invoke-Expression $scpCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Success "Download concluído!"
        Write-Host "📂 Resultados em: $localDir"
        return $true
    } else {
        Write-Error "Falha no download"
        return $false
    }
}

# ============================================================================
# MAIN
# ============================================================================

Write-Header "GOOGLE COLAB WORKFLOW via SSH"

Write-Host "🔧 Configuração:" -ForegroundColor Cyan
Write-Host "   SSH Host: $SSHHost"
Write-Host "   Action:   $Action"
Write-Host ""

# Executa ação selecionada
switch ($Action) {
    'sync' {
        $result = Invoke-Sync
        exit $(if ($result) { 0 } else { 1 })
    }

    'setup' {
        $result = Invoke-Setup
        exit $(if ($result) { 0 } else { 1 })
    }

    'run' {
        $result = Invoke-Run
        exit $(if ($result) { 0 } else { 1 })
    }

    'download' {
        $result = Invoke-Download
        exit $(if ($result) { 0 } else { 1 })
    }

    'all' {
        # Workflow completo: sync -> setup -> run
        Write-Header "WORKFLOW COMPLETO"

        # Sync
        if (-not (Invoke-Sync)) {
            Write-Error "Sync falhou. Abortando."
            exit 1
        }

        Start-Sleep -Seconds 2

        # Setup
        if (-not (Invoke-Setup)) {
            Write-Error "Setup falhou. Abortando."
            Write-Warning "Execute manualmente no notebook Colab:"
            Write-Warning "  from google.colab import drive"
            Write-Warning "  drive.mount('/content/drive')"
            exit 1
        }

        Start-Sleep -Seconds 2

        # Run
        if (-not (Invoke-Run)) {
            Write-Error "Execução falhou."
            exit 1
        }

        # Resumo final
        Write-Header "WORKFLOW CONCLUÍDO COM SUCESSO!"

        Write-Host "✅ Todas as etapas executadas:" -ForegroundColor Green
        Write-Host "   1. Sync   ✅"
        Write-Host "   2. Setup  ✅"
        Write-Host "   3. Run    ✅"
        Write-Host ""
        Write-Host "📊 Resultados salvos no Google Drive" -ForegroundColor Cyan
        Write-Host "   Acesse: drive.google.com → MyDrive/DSL-AG-hybrid/experiments/"
        Write-Host ""
        Write-Host "💡 Para baixar resultados:" -ForegroundColor Yellow
        Write-Host "   .\colab_workflow.ps1 -SSHHost $SSHHost -Action download"
        Write-Host ""

        exit 0
    }
}
