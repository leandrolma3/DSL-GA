@echo off
REM Script Completo: Instala Dependencias e Executa Teste
REM Execute este arquivo para configurar e testar o sistema

cd /d "%~dp0"

echo ======================================================================
echo  INSTALACAO E TESTE - Sistema de Comparacao GBML vs River
echo ======================================================================
echo.

echo [1/4] Instalando dependencias criticas...
echo   Isso pode demorar alguns minutos...
echo.

pip install pandas scikit-learn river seaborn python-Levenshtein xgboost --quiet --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na instalacao de dependencias!
    echo Por favor, execute manualmente:
    echo   pip install pandas scikit-learn river seaborn
    pause
    exit /b 1
)

echo   [OK] Dependencias instaladas!
echo.

echo [2/4] Validando instalacao...
python quick_test.py

if errorlevel 1 (
    echo.
    echo [ERRO] Validacao falhou!
    pause
    exit /b 1
)

echo.
echo [3/4] Executando teste rapido (apenas River - 30 segundos)...
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2 --chunk-size 500 --no-gbml --models HAT

if errorlevel 1 (
    echo.
    echo [ERRO] Teste River falhou!
    echo Verifique os logs acima
    pause
    exit /b 1
)

echo.
echo [4/4] Executando teste completo (GBML + River - 2-3 minutos)...
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2 --chunk-size 500 --models HAT

echo.
echo ======================================================================
echo  TESTE CONCLUIDO COM SUCESSO!
echo ======================================================================
echo.
echo Resultados salvos em: comparison_results\
echo.
pause
