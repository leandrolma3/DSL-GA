@echo off
REM Script para upload de arquivos via SCP
REM Autor: Claude Code
REM Data: 2025-10-18

set SSH_HOST=%1

if "%SSH_HOST%"=="" (
    echo.
    echo ERRO: Host SSH nao fornecido
    echo.
    echo Uso: upload_files.bat HOST
    echo Exemplo: upload_files.bat logged-minerals-axis-infrastructure.trycloudflare.com
    echo.
    exit /b 1
)

echo.
echo ========================================================================
echo   UPLOAD DE ARQUIVOS PARA GOOGLE COLAB VIA SSH
echo ========================================================================
echo.
echo Host SSH: %SSH_HOST%
echo.
echo NOTA: Voce precisara digitar a senha 'root' varias vezes.
echo.
pause

echo.
echo Criando diretorio remoto...
ssh %SSH_HOST% "mkdir -p /root/DSL-AG-hybrid"

echo.
echo Fazendo upload dos arquivos...
echo.

echo [1/11] main.py
scp -q main.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [2/11] ga.py
scp -q ga.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [3/11] ga_operators.py
scp -q ga_operators.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [4/11] config.yaml
scp -q config.yaml %SSH_HOST%:/root/DSL-AG-hybrid/

echo [5/11] compare_gbml_vs_river.py
scp -q compare_gbml_vs_river.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [6/11] hill_climbing_v2.py
scp -q hill_climbing_v2.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [7/11] intelligent_hc_strategies.py
scp -q intelligent_hc_strategies.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [8/11] dt_rule_extraction.py
scp -q dt_rule_extraction.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [9/11] early_stopping.py
scp -q early_stopping.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [10/11] utils.py
scp -q utils.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo [11/11] setup_colab_remote.py
scp -q setup_colab_remote.py %SSH_HOST%:/root/DSL-AG-hybrid/

echo.
echo ========================================================================
echo   UPLOAD CONCLUIDO
echo ========================================================================
echo.
echo Proximos passos:
echo.
echo 1. Conecte via SSH:
echo    ssh %SSH_HOST%
echo.
echo 2. Execute setup:
echo    cd /root/DSL-AG-hybrid
echo    python3 setup_colab_remote.py
echo.
echo 3. Execute experimento:
echo    /root/run_experiment.sh python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 3
echo.

pause
