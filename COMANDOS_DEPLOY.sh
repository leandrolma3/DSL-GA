#!/bin/bash
################################################################################
# COMANDOS DE DEPLOY - FASE 1-NOVO: Correções Drift SEVERE
################################################################################
# Data: 2025-10-23
# Objetivo: Sincronizar correções (memory parcial, herança 20%, seeding 85%)
################################################################################

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0;0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         DEPLOY - FASE 1-NOVO: CORREÇÕES DRIFT SEVERE                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Correções implementadas:${NC}"
echo -e "  ✅ Memory parcial (top 10%) em SEVERE"
echo -e "  ✅ Herança mínima 20% em SEVERE"
echo -e "  ✅ Seeding intensivo 85% em SEVERE"
echo -e "  ✅ Hill Climbing 18 variantes (P2B corrigido)"
echo ""

# 1. DEFINA O SSH HOST
# Substitua pelo seu host SSH Cloudflare
SSH_HOST="YOUR_SSH_HOST.trycloudflare.com"

echo -e "${YELLOW}📝 Configuração:${NC}"
echo -e "   SSH Host: ${BLUE}$SSH_HOST${NC}"
echo ""
echo -e "${YELLOW}Pressione ENTER para continuar ou Ctrl+C para cancelar${NC}"
read

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASSO 1: SINCRONIZAR ARQUIVOS MODIFICADOS${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Arquivos a sincronizar
FILES=(
    "main.py"
    "ga.py"
    "hill_climbing_v2.py"
    "setup_colab_remote.py"
    "setup_ssh_with_drive.py"
)

echo -e "${BLUE}Sincronizando ${#FILES[@]} arquivos...${NC}"
echo ""

SUCCESS=0
FAILED=0

for file in "${FILES[@]}"; do
    echo -ne "${BLUE}📤 $file...${NC} "

    if scp -q "$file" "$SSH_HOST:/root/DSL-AG-hybrid/" 2>/dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${YELLOW}❌ FALHOU${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Sincronização concluída:${NC} ✅ $SUCCESS OK | ❌ $FAILED FALHAS"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Alguns arquivos falharam. Verifique a conexão SSH.${NC}"
    echo ""
    exit 1
fi

echo -e "${YELLOW}Pressione ENTER para continuar com o teste${NC}"
read

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASSO 2: EXPERIMENTO COMPLETO (6 chunks)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}📝 Conectando via SSH...${NC}"
echo ""

# Executa teste isolado chunk 4→5
ssh -t "$SSH_HOST" << 'EOF'
cd /root/DSL-AG-hybrid

echo "════════════════════════════════════════════════════════════"
echo "🧪 EXPERIMENTO COMPLETO: 6 Chunks (Validação Fase 1-Novo)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏱️ Tempo estimado: ~11-12h"
echo "🎯 Objetivo: Validar correções de drift SEVERE (chunk 4→5 ≥ 55%)"
echo ""
echo "📊 Metas:"
echo "   • Chunk 4→5: ≥ 55% (P1+P2 foi 39.02%)"
echo "   • Avg G-mean: ≥ 81% (P1+P2 foi 78.07%)"
echo "   • Drift detection: 100% (funciona)"
echo ""
echo "Comando: python main.py"
echo ""
echo "Pressione ENTER para iniciar ou Ctrl+C para cancelar"
read

# Cria timestamp para log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="fase1_novo_$TIMESTAMP.log"

echo ""
echo "📁 Log: $LOG_FILE"
echo ""

# Executa teste com logging
nohup python main.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "✅ Teste iniciado em background (PID: $PID)"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      COMANDOS ÚTEIS                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Monitorar progresso:"
echo "   tail -f $LOG_FILE | grep -E 'Chunk.*Results|DRIFT|HC.*aprovadas'"
echo ""
echo "🔍 Buscar drift detection:"
echo "   grep -E '🔴 SEVERE|🟡 MODERATE|🟢 MILD|✓ STABLE' $LOG_FILE"
echo ""
echo "🔍 Validar correções SEVERE:"
echo "   grep -E 'Memory REDUCED|Inheritance REDUCED to 20%|SEVERE DRIFT DETECTED: Seeding' $LOG_FILE"
echo ""
echo "📈 Ver taxa HC:"
echo "   grep 'taxa de aprovação' $LOG_FILE"
echo ""
echo "🛑 Parar execução:"
echo "   kill $PID"
echo ""
echo "Pressione ENTER para monitorar (Ctrl+C para sair do monitoramento)"
read

# Monitora log em tempo real
tail -f "$LOG_FILE" | grep --line-buffered -E "Chunk.*Results|DRIFT|HC.*aprovadas|EARLY STOPPING"
EOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}TESTE CONCLUÍDO${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Próximos passos:${NC}"
echo "  1. Verificar log do experimento"
echo "  2. Validar correções SEVERE (memory, herança, seeding)"
echo "  3. Comparar chunk 4→5: esperado ≥ 55% (P1+P2 foi 39.02%)"
echo "  4. Se OK (≥55%): Prosseguir para Fase 2-Novo (melhorias HC)"
echo "  5. Se NOK (<50%): Revisar hipótese e ajustar parâmetros"
echo ""
