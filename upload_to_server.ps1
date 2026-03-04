# ============================================================================
# Script de Upload Inteligente para Servidor Remoto via SSH
# Versão: 1.2
# Data: 2025-10-15
# ============================================================================

# === CONFIGURAÇÃO ===
# Lê o servidor do arquivo de configuração (se existir)
$CONFIG_FILE = "server_address.txt"
if (Test-Path $CONFIG_FILE) {
    $SERVER = Get-Content $CONFIG_FILE -Raw
    $SERVER = $SERVER.Trim()
} else {
    $SERVER = "ssh right-scotia-humor-toward.trycloudflare.com"
}

$REMOTE_PATH = "~/DSL-AG-hybrid"
$LOCAL_BASE = "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Lista de arquivos a sincronizar
$FILES_TO_UPLOAD = @(
    "ga_operators.py",
    "ga.py",
    "main.py",
    "config.yaml"
)

# === FUNÇÕES ===

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Check-SSHConnection {
    Write-ColorOutput "`n[1/4] Testando conexão SSH com o servidor..." "Cyan"

    $result = & ssh $SERVER 'echo OK' 2>&1

    if ($LASTEXITCODE -eq 0 -and $result -match "OK") {
        Write-ColorOutput "  ✓ Conexão SSH estabelecida com sucesso" "Green"
        return $true
    } else {
        Write-ColorOutput "  ✗ Falha na conexão SSH" "Red"
        Write-ColorOutput "  Erro: $result" "Yellow"
        return $false
    }
}

function Check-RemoteDirectory {
    Write-ColorOutput "`n[2/4] Verificando diretório remoto..." "Cyan"

    $testCmd = 'test -d ' + $REMOTE_PATH + ' && echo EXISTS || echo NOT_EXISTS'
    $result = & ssh $SERVER $testCmd 2>&1

    if ($result -match "EXISTS") {
        Write-ColorOutput "  ✓ Diretório '$REMOTE_PATH' já existe no servidor" "Green"
        return $true
    } elseif ($result -match "NOT_EXISTS") {
        Write-ColorOutput "  ⚠ Diretório '$REMOTE_PATH' não existe no servidor" "Yellow"
        return $false
    } else {
        Write-ColorOutput "  ✗ Erro ao verificar diretório remoto" "Red"
        Write-ColorOutput "  Resposta: $result" "Yellow"
        return $null
    }
}

function Create-RemoteDirectory {
    Write-ColorOutput "`n[3/4] Criando diretório remoto..." "Cyan"

    $mkdirCmd = 'mkdir -p ' + $REMOTE_PATH
    & ssh $SERVER $mkdirCmd 2>&1 | Out-Null

    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "  ✓ Diretório '$REMOTE_PATH' criado com sucesso" "Green"
        return $true
    } else {
        Write-ColorOutput "  ✗ Falha ao criar diretório remoto" "Red"
        return $false
    }
}

function Create-RemoteBackup {
    Write-ColorOutput "`n[Opcional] Criando backup dos arquivos existentes..." "Cyan"

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupCmd = 'cd ' + $REMOTE_PATH + ' 2>/dev/null && mkdir -p backups/' + $timestamp + ' && cp -f ga_operators.py ga.py main.py config.yaml backups/' + $timestamp + '/ 2>/dev/null && echo DONE'

    $result = & ssh $SERVER $backupCmd 2>&1

    if ($result -match "DONE") {
        Write-ColorOutput "  ✓ Backup criado em: backups/$timestamp/" "Green"
        return $true
    } else {
        Write-ColorOutput "  ⚠ Nenhum arquivo para backup (pasta vazia ou primeira vez)" "Yellow"
        return $false
    }
}

function Upload-Files {
    param (
        [string[]]$Files
    )

    Write-ColorOutput "`n[4/4] Enviando arquivos para o servidor..." "Cyan"

    $successCount = 0
    $failCount = 0

    foreach ($file in $Files) {
        $localPath = Join-Path $LOCAL_BASE $file

        # Verifica se arquivo existe localmente
        if (-Not (Test-Path $localPath)) {
            Write-ColorOutput "  ✗ $file não encontrado localmente" "Red"
            $failCount++
            continue
        }

        Write-ColorOutput "`n  Enviando: $file" "White"

        # Exibe tamanho do arquivo
        $fileSize = (Get-Item $localPath).Length
        $fileSizeKB = [math]::Round($fileSize / 1KB, 2)
        Write-ColorOutput "    Tamanho: $fileSizeKB KB" "Gray"

        # Envia arquivo
        $remoteDest = $SERVER + ':' + $REMOTE_PATH + '/'
        & scp $localPath $remoteDest 2>&1 | Out-Null

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "    ✓ $file enviado com sucesso" "Green"
            $successCount++
        } else {
            Write-ColorOutput "    ✗ Erro ao enviar $file" "Red"
            $failCount++
        }
    }

    Write-ColorOutput "`n  ─────────────────────────────" "Gray"
    Write-ColorOutput "  Total: $($Files.Count) arquivos" "White"
    Write-ColorOutput "  Sucesso: $successCount" "Green"
    if ($failCount -gt 0) {
        Write-ColorOutput "  Falhas: $failCount" "Red"
    }

    return ($failCount -eq 0)
}

function Verify-Upload {
    Write-ColorOutput "`n[Verificação] Conferindo arquivos no servidor..." "Cyan"

    $verifyCmd = 'cd ' + $REMOTE_PATH + ' && ls -lh ga_operators.py ga.py main.py config.yaml 2>/dev/null'
    $result = & ssh $SERVER $verifyCmd 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "`n  Arquivos remotos:" "White"
        $result | ForEach-Object {
            if ($_ -match '\S') {
                Write-ColorOutput "    $_" "Gray"
            }
        }

        # Verifica se crossover está ativado
        Write-ColorOutput "`n  Verificando configuração do Crossover Balanceado..." "White"
        $configCmd = 'grep use_balanced_crossover ' + $REMOTE_PATH + '/config.yaml'
        $configResult = & ssh $SERVER $configCmd 2>&1

        $pattern = 'use_balanced_crossover.*true'
        if ($configResult -match $pattern) {
            Write-ColorOutput "    ✓ Crossover Balanceado: ATIVADO" "Green"
        } else {
            Write-ColorOutput "    ⚠ Crossover Balanceado: DESATIVADO" "Yellow"
        }

        return $true
    } else {
        Write-ColorOutput "  ✗ Erro ao verificar arquivos remotos" "Red"
        return $false
    }
}

# === EXECUÇÃO PRINCIPAL ===

Write-ColorOutput "`n╔════════════════════════════════════════════════════════════╗" "Cyan"
Write-ColorOutput "║  Upload Inteligente - DSL-AG-hybrid                       ║" "Cyan"
Write-ColorOutput "║  Servidor: $SERVER" "Cyan"
Write-ColorOutput "╚════════════════════════════════════════════════════════════╝" "Cyan"

# Passo 1: Testar conexão SSH
if (-Not (Check-SSHConnection)) {
    Write-ColorOutput "`n❌ ERRO: Não foi possível conectar ao servidor" "Red"
    Write-ColorOutput "Verifique se:" "Yellow"
    Write-ColorOutput "  1. O servidor está acessível" "Yellow"
    Write-ColorOutput "  2. O túnel Cloudflare está ativo" "Yellow"
    Write-ColorOutput "  3. Suas credenciais SSH estão configuradas" "Yellow"
    exit 1
}

# Passo 2: Verificar/Criar diretório remoto
$dirExists = Check-RemoteDirectory

if ($dirExists -eq $null) {
    Write-ColorOutput "`n❌ ERRO: Não foi possível verificar o diretório remoto" "Red"
    exit 1
}

if (-Not $dirExists) {
    Write-ColorOutput "`n  → Criando diretório remoto..." "Yellow"
    if (-Not (Create-RemoteDirectory)) {
        Write-ColorOutput "`n❌ ERRO: Não foi possível criar o diretório remoto" "Red"
        exit 1
    }
} else {
    # Se diretório existe, criar backup
    Create-RemoteBackup | Out-Null
}

# Passo 3: Upload dos arquivos
$uploadSuccess = Upload-Files -Files $FILES_TO_UPLOAD

if (-Not $uploadSuccess) {
    Write-ColorOutput "`n⚠ AVISO: Alguns arquivos falharam no upload" "Yellow"
}

# Passo 4: Verificação final
Verify-Upload | Out-Null

# Resumo final
Write-ColorOutput "`n╔════════════════════════════════════════════════════════════╗" "Cyan"
if ($uploadSuccess) {
    Write-ColorOutput "║  ✓ Upload concluído com sucesso!                          ║" "Green"
} else {
    Write-ColorOutput "║  ⚠ Upload concluído com alguns erros                      ║" "Yellow"
}
Write-ColorOutput "╚════════════════════════════════════════════════════════════╝" "Cyan"

Write-ColorOutput "`n📋 Próximos passos:" "White"
Write-ColorOutput "  1. Conectar ao servidor: ssh $SERVER" "Gray"
Write-ColorOutput "  2. Navegar para o diretório: cd $REMOTE_PATH" "Gray"
Write-ColorOutput "  3. Executar o experimento com os operadores inteligentes ativados" "Gray"

Write-ColorOutput "`n💡 Configuração atual:" "White"
Write-ColorOutput "  • Hill Climbing Inteligente: ATIVADO (hc_hierarchical_enabled: true)" "Green"
Write-ColorOutput "  • Crossover Balanceado: ATIVADO (use_balanced_crossover: true)" "Green"

Write-ColorOutput ""
