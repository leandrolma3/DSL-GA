# Upload SSH - Versao Simplificada e Robusta
$Server = "frozen-about-ball-indicating.trycloudflare.com"
$User = "root"
$RemotePath = "/content/DSL-AG-hybrid"
$LocalBase = $PSScriptRoot

Write-Host ""
Write-Host "========================================"
Write-Host "Upload SSH - DSL-AG-hybrid"
Write-Host "Servidor: $User@$Server"
Write-Host "Destino: $RemotePath"
Write-Host "========================================"
Write-Host ""

# Verificar se pasta remota existe
Write-Host "[1/4] Verificando pasta remota..."
Write-Host "Digite a senha:"
$checkCmd = "test -d $RemotePath && echo EXISTS || echo NOT_EXISTS"
$checkResult = & ssh -o StrictHostKeyChecking=no "$User@$Server" $checkCmd 2>&1

Write-Host ""
Write-Host "Resultado: $checkResult"
Write-Host ""

$folderExists = $false
if ($checkResult -match "EXISTS") {
    Write-Host "OK - Pasta ja existe"
    Write-Host ""
    $folderExists = $true
} else {
    Write-Host "AVISO - Pasta nao existe (primeira vez)"
    Write-Host ""
    $folderExists = $false
}

# Se pasta nao existe, copiar tudo
if (-not $folderExists) {
    Write-Host "[2/4] PRIMEIRA VEZ: Copiando TODA a pasta..."
    Write-Host "Digite a senha:"
    Write-Host ""
    Write-Host "Isso pode levar alguns minutos..."
    Write-Host ""

    # Copiar toda a pasta usando scp recursivo
    & scp -o StrictHostKeyChecking=no -r $LocalBase "$User@${Server}:/content/"

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "OK - Pasta completa copiada"
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "ERRO - Falha ao copiar pasta (codigo: $LASTEXITCODE)"
        Write-Host ""
        pause
        exit 1
    }
} else {
    Write-Host "[2/4] Pasta ja existe, pulando copia completa..."
    Write-Host ""
}

# Garantir que diretorio existe (caso tenha havido erro na verificacao)
Write-Host "[3/4] Garantindo que diretorio existe..."
Write-Host "Digite a senha:"
$mkdirCmd = "mkdir -p $RemotePath"
& ssh -o StrictHostKeyChecking=no "$User@$Server" $mkdirCmd 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Diretorio pronto"
    Write-Host ""
} else {
    Write-Host "AVISO - Prosseguindo mesmo assim"
    Write-Host ""
}

# Atualizar apenas os arquivos importantes
Write-Host "[4/4] Sincronizando arquivos atualizados..."
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
$success = 0
$num = 1
$total = $files.Count

foreach ($f in $files) {
    $localFile = Join-Path $LocalBase $f

    if (Test-Path $localFile) {
        $size = [math]::Round((Get-Item $localFile).Length / 1KB, 1)
        Write-Host "  [$num/$total] Enviando: $f ($size KB)"
        Write-Host "      Digite a senha:"

        & scp -o StrictHostKeyChecking=no $localFile "$User@${Server}:${RemotePath}/"

        if ($LASTEXITCODE -eq 0) {
            Write-Host "      OK - Enviado"
            Write-Host ""
            $success = $success + 1
        } else {
            Write-Host "      ERRO - Falhou (codigo: $LASTEXITCODE)"
            Write-Host ""
        }
    } else {
        Write-Host "  [$num/$total] $f - NAO ENCONTRADO"
        Write-Host ""
    }

    $num = $num + 1
}

Write-Host "========================================"
Write-Host "RESULTADO: $success/$total arquivos enviados"
Write-Host "========================================"
Write-Host ""
Write-Host "Configuracao:"
Write-Host "  HC Inteligente: ATIVADO"
Write-Host "  Crossover Balanceado: ATIVADO"
Write-Host ""
Write-Host "Localizacao no servidor:"
Write-Host "  $RemotePath"
Write-Host ""
Write-Host "=========================================="
Write-Host "PROXIMOS PASSOS:"
Write-Host "=========================================="
Write-Host ""
Write-Host "1. MONTAR GOOGLE DRIVE (se ainda nao fez):"
Write-Host "   - Abra notebook Colab no navegador"
Write-Host "   - Execute: from google.colab import drive"
Write-Host "              drive.mount('/content/drive')"
Write-Host ""
Write-Host "2. CONECTAR VIA SSH:"
Write-Host "   ssh $User@$Server"
Write-Host ""
Write-Host "3. SETUP DO DRIVE:"
Write-Host "   cd $RemotePath"
Write-Host "   python3 setup_colab_remote.py"
Write-Host ""
Write-Host "4. EXECUTAR EXPERIMENTO:"
Write-Host "   /root/run_experiment.sh python3 compare_gbml_vs_river.py \"
Write-Host "       --stream RBF_Abrupt_Severe \"
Write-Host "       --chunks 1"
Write-Host ""
Write-Host "5. VER RESULTADOS:"
Write-Host "   tail -50 /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/*.log"
Write-Host ""

pause
