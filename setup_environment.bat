@echo off
REM Script de Configuracao do Ambiente - GBML vs River Comparison
REM Execute este script para configurar o ambiente completo

echo ======================================================================
echo  CONFIGURACAO DO AMBIENTE - GBML vs River Comparison
echo ======================================================================

cd /d "%~dp0"

REM Verifica se o ambiente existe
if not exist "venv_comparison\" (
    echo [1/3] Criando ambiente virtual...
    python -m venv venv_comparison
    if errorlevel 1 (
        echo ERRO: Falha ao criar ambiente virtual
        pause
        exit /b 1
    )
    echo     Ambiente virtual criado com sucesso!
) else (
    echo [1/3] Ambiente virtual ja existe
)

echo.
echo [2/3] Ativando ambiente virtual...
call venv_comparison\Scripts\activate.bat

echo.
echo [3/3] Instalando dependencias...
echo     Atualizando pip...
python -m pip install --upgrade pip --quiet

echo     Instalando pacotes principais...
pip install numpy pandas scikit-learn pyyaml --quiet
echo       - numpy, pandas, scikit-learn, pyyaml OK

echo     Instalando River (stream learning)...
pip install river --quiet
echo       - river OK

echo     Instalando visualizacao...
pip install matplotlib seaborn --quiet
echo       - matplotlib, seaborn OK

echo     Instalando utilitarios...
pip install python-Levenshtein xgboost --quiet
echo       - Levenshtein, xgboost OK

echo.
echo ======================================================================
echo  INSTALACAO CONCLUIDA COM SUCESSO!
echo ======================================================================
echo.
echo O ambiente esta ativo. Voce pode agora executar:
echo   python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 3
echo.
echo Para reativar o ambiente no futuro:
echo   venv_comparison\Scripts\activate
echo.
pause
