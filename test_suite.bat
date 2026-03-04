@echo off
REM Suite de Testes - GBML vs River (Após Correções)
REM Execute este script para validar todas as correções

cd /d "%~dp0"

echo ======================================================================
echo  SUITE DE TESTES - GBML vs River (Pos-Correcoes)
echo ======================================================================
echo.

echo [Teste 1/4] Validando HAT (River)...
echo.
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2 --chunk-size 1000 --models HAT --no-gbml

if errorlevel 1 (
    echo.
    echo [ERRO] HAT falhou! Verifique os logs acima.
    pause
    exit /b 1
)

echo.
echo [OK] HAT passou!
echo.
echo ======================================================================
echo.

echo [Teste 2/4] Validando ARF (River)...
echo.
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2 --chunk-size 1000 --models ARF --no-gbml

if errorlevel 1 (
    echo.
    echo [ERRO] ARF falhou! Verifique os logs acima.
    pause
    exit /b 1
)

echo.
echo [OK] ARF passou!
echo.
echo ======================================================================
echo.

echo [Teste 3/4] Validando GBML isolado...
echo.
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2 --chunk-size 500 --no-river

if errorlevel 1 (
    echo.
    echo [ERRO] GBML falhou! Verifique os logs acima.
    pause
    exit /b 1
)

echo.
echo [OK] GBML passou!
echo.
echo ======================================================================
echo.

echo [Teste 4/4] Validando comparacao completa (GBML + HAT + ARF + SRP)...
echo.
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2 --chunk-size 500 --models HAT ARF SRP

if errorlevel 1 (
    echo.
    echo [ERRO] Comparacao completa falhou!
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  TODOS OS TESTES PASSARAM!
echo ======================================================================
echo.
echo Resultados salvos em: comparison_results\
echo.
pause
