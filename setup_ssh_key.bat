@echo off
REM ============================================================================
REM Configurar Chave SSH para Upload Sem Senha
REM Execute este script APENAS UMA VEZ
REM ============================================================================

echo.
echo ========================================
echo   CONFIGURACAO DE CHAVE SSH
echo ========================================
echo.
echo Este script vai:
echo   1. Gerar uma chave SSH (se nao existir)
echo   2. Copiar a chave para o servidor
echo   3. Testar conexao sem senha
echo.
echo IMPORTANTE: Voce precisara digitar a SENHA DO SERVIDOR
echo.
pause

powershell -ExecutionPolicy Bypass -File "%~dp0setup_ssh_key.ps1"
