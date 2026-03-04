#!/bin/bash
################################################################################
# sync_to_colab.sh
################################################################################
# Script para sincronizar arquivos locais para Google Colab via SSH
#
# Uso:
#   ./sync_to_colab.sh <ssh-host>
#
# Exemplo:
#   ./sync_to_colab.sh logged-minerals-axis-infrastructure.trycloudflare.com
#
# Autor: Claude Code
# Data: 2025-10-18
################################################################################

set -e  # Para na primeira falha

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verifica argumentos
if [ -z "$1" ]; then
    echo -e "${RED}❌ ERRO: SSH host não fornecido${NC}"
    echo ""
    echo "Uso: $0 <ssh-host>"
    echo "Exemplo: $0 logged-minerals-axis-infrastructure.trycloudflare.com"
    exit 1
fi

SSH_HOST="$1"
PROJECT_NAME="DSL-AG-hybrid"
REMOTE_DIR="/root/${PROJECT_NAME}"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              SYNC TO GOOGLE COLAB via SSH                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📡 SSH Host:${NC} $SSH_HOST"
echo -e "${YELLOW}📂 Destino:${NC} $REMOTE_DIR"
echo ""

# Arquivos a serem sincronizados
FILES_TO_SYNC=(
    "main.py"
    "ga.py"
    "ga_operators.py"
    "config.yaml"
    "compare_gbml_vs_river.py"
    "hill_climbing_v2.py"
    "intelligent_hc_strategies.py"
    "dt_rule_extraction.py"
    "colab_drive_setup.py"
    "early_stopping.py"
    "utils.py"
    "requirements.txt"
)

# Contador de arquivos
TOTAL=${#FILES_TO_SYNC[@]}
COUNT=0
SUCCESS=0
FAILED=0

echo -e "${BLUE}📦 Sincronizando $TOTAL arquivos...${NC}"
echo ""

# Cria diretório remoto se não existir
echo -e "${YELLOW}🔧 Criando diretório remoto...${NC}"
ssh "$SSH_HOST" "mkdir -p $REMOTE_DIR" 2>/dev/null || {
    echo -e "${RED}❌ Erro ao criar diretório remoto${NC}"
    exit 1
}

# Sincroniza cada arquivo
for file in "${FILES_TO_SYNC[@]}"; do
    COUNT=$((COUNT + 1))

    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}⚠️  [$COUNT/$TOTAL] $file - SKIP (não existe localmente)${NC}"
        continue
    fi

    echo -ne "${BLUE}📤 [$COUNT/$TOTAL] $file...${NC} "

    if scp -q "$file" "$SSH_HOST:$REMOTE_DIR/" 2>/dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}❌ FALHOU${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                       RESUMO DA SINCRONIZAÇÃO                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}✅ Sucesso:${NC} $SUCCESS arquivos"
echo -e "${RED}❌ Falhas:${NC}  $FAILED arquivos"
echo -e "${YELLOW}📂 Total:${NC}   $TOTAL arquivos"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 SINCRONIZAÇÃO CONCLUÍDA COM SUCESSO!${NC}"
    echo ""
    echo -e "${BLUE}Próximos passos:${NC}"
    echo -e "  1. Conecte via SSH: ${YELLOW}ssh $SSH_HOST${NC}"
    echo -e "  2. Execute setup:   ${YELLOW}cd $REMOTE_DIR && python setup_colab_remote.py${NC}"
    echo -e "  3. Execute teste:   ${YELLOW}python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 1${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  SINCRONIZAÇÃO CONCLUÍDA COM ERROS${NC}"
    echo ""
    exit 1
fi
