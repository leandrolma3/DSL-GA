# 📤 Sistema de Upload Inteligente para Servidor SSH

## 📋 Arquivos do Sistema

- **`upload_to_server.ps1`** - Script principal de upload
- **`server_config.ps1`** - Utilitário para gerenciar configuração do servidor
- **`server_address.txt`** - Arquivo de configuração (criado automaticamente)

---

## 🚀 Uso Básico

### 1. Primeiro Upload (Configuração Padrão)

```powershell
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
.\upload_to_server.ps1
```

**O script irá automaticamente:**
- ✓ Testar conexão SSH
- ✓ Verificar se a pasta remota existe
- ✓ Criar a pasta se necessário
- ✓ Fazer backup dos arquivos existentes
- ✓ Enviar os 4 arquivos modificados
- ✓ Verificar se o upload foi bem-sucedido

---

## ⚙️ Mudando o Servidor

### Opção A: Usando o Utilitário Interativo

```powershell
.\server_config.ps1
```

**Menu interativo:**
```
╔════════════════════════════════════════════════════════╗
║        Configuração de Servidor SSH                   ║
╚════════════════════════════════════════════════════════╝

1. Ver servidor atual
2. Alterar servidor
3. Testar conexão
4. Restaurar servidor padrão
5. Sair
```

**Exemplo de uso:**
1. Escolha opção `2` (Alterar servidor)
2. Digite o novo endereço: `ssh new-server.trycloudflare.com`
3. Escolha opção `3` (Testar conexão) para validar
4. Saia com opção `5`

### Opção B: Edição Manual

Crie/edite o arquivo `server_address.txt`:

```
ssh new-tunnel-address.trycloudflare.com
```

---

## 📦 Arquivos Sincronizados

O sistema sincroniza automaticamente os seguintes arquivos:

| Arquivo | Descrição |
|---------|-----------|
| `ga_operators.py` | Operadores genéticos (Crossover Balanceado) |
| `ga.py` | Loop principal do GA |
| `main.py` | Ponto de entrada e configuração |
| `config.yaml` | Configuração do sistema |

---

## 🔍 Funcionalidades Avançadas

### Backup Automático

Antes de sobrescrever arquivos existentes, o script cria backup:

```
~/DSL-AG-hybrid/backups/20251015_143022/
  ├── ga_operators.py
  ├── ga.py
  ├── main.py
  └── config.yaml
```

### Verificação Pós-Upload

O script verifica automaticamente:
- ✓ Tamanho dos arquivos remotos
- ✓ Status do Crossover Balanceado (`use_balanced_crossover`)
- ✓ Listagem dos arquivos no servidor

---

## 🛠️ Solução de Problemas

### Erro: "Conexão SSH falhou"

**Possíveis causas:**
1. Túnel Cloudflare não está ativo
2. Endereço do servidor incorreto
3. Credenciais SSH não configuradas

**Solução:**
```powershell
# Testar conexão manualmente
ssh "ssh right-scotia-humor-toward.trycloudflare.com" "echo OK"

# Verificar configuração do servidor
.\server_config.ps1
```

---

### Erro: "Arquivo não encontrado localmente"

**Causa:** Arquivo não existe no diretório local

**Solução:**
```powershell
# Verificar arquivos locais
ls ga_operators.py, ga.py, main.py, config.yaml
```

---

### Erro: "Permissão negada"

**Causa:** Pasta remota sem permissão de escrita

**Solução:**
```bash
# No servidor remoto
ssh "seu-servidor.com"
chmod -R u+w ~/DSL-AG-hybrid
```

---

## 📊 Exemplo de Saída

```
╔════════════════════════════════════════════════════════════╗
║  Upload Inteligente - DSL-AG-hybrid                       ║
║  Servidor: ssh right-scotia-humor-toward.trycloudflare.com║
╚════════════════════════════════════════════════════════════╝

[1/4] Testando conexão SSH com o servidor...
  ✓ Conexão SSH estabelecida com sucesso

[2/4] Verificando diretório remoto...
  ✓ Diretório '~/DSL-AG-hybrid' já existe no servidor

[Opcional] Criando backup dos arquivos existentes...
  ✓ Backup criado em: backups/20251015_143022/

[4/4] Enviando arquivos para o servidor...

  Enviando: ga_operators.py
    Tamanho: 78.45 KB
    ✓ ga_operators.py enviado com sucesso

  Enviando: ga.py
    Tamanho: 45.12 KB
    ✓ ga.py enviado com sucesso

  Enviando: main.py
    Tamanho: 52.30 KB
    ✓ main.py enviado com sucesso

  Enviando: config.yaml
    Tamanho: 18.76 KB
    ✓ config.yaml enviado com sucesso

  ─────────────────────────────
  Total: 4 arquivos
  Sucesso: 4

[Verificação] Conferindo arquivos no servidor...

  Arquivos remotos:
    ga_operators.py 78K
    ga.py 46K
    main.py 53K
    config.yaml 19K

  Verificando configuração do Crossover Balanceado...
    ✓ Crossover Balanceado: ATIVADO

╔════════════════════════════════════════════════════════════╗
║  ✓ Upload concluído com sucesso!                          ║
╚════════════════════════════════════════════════════════════╝

📋 Próximos passos:
  1. Conectar ao servidor: ssh ssh right-scotia-humor-toward.trycloudflare.com
  2. Navegar para o diretório: cd ~/DSL-AG-hybrid
  3. Executar o experimento com os operadores inteligentes ativados

💡 Configuração atual:
  • Hill Climbing Inteligente: ATIVADO (hc_hierarchical_enabled: true)
  • Crossover Balanceado: ATIVADO (use_balanced_crossover: true)
```

---

## 🔐 Configuração SSH (Primeira Vez)

Se você ainda não tem SSH configurado:

### Windows (PowerShell como Administrador)

```powershell
# Instalar OpenSSH Client
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# Gerar chave SSH (se não tiver)
ssh-keygen -t rsa -b 4096

# Copiar chave pública para o servidor
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh seu-servidor.com "cat >> ~/.ssh/authorized_keys"
```

---

## 📝 Notas Importantes

1. **Servidor Padrão**: `ssh right-scotia-humor-toward.trycloudflare.com`
2. **Diretório Remoto**: `~/DSL-AG-hybrid` (criado automaticamente)
3. **Backups**: Mantidos em `~/DSL-AG-hybrid/backups/` (com timestamp)
4. **Configuração Atual**:
   - Hill Climbing Inteligente: **ATIVADO** ✅
   - Crossover Balanceado: **ATIVADO** ✅

---

## 🎯 Próximos Passos Após Upload

1. **Conectar ao servidor:**
   ```bash
   ssh "ssh right-scotia-humor-toward.trycloudflare.com"
   ```

2. **Navegar para o diretório:**
   ```bash
   cd ~/DSL-AG-hybrid
   ```

3. **Verificar configuração:**
   ```bash
   grep "use_balanced_crossover\|hc_hierarchical_enabled" config.yaml
   ```

4. **Executar experimento de teste (1 chunk):**
   ```bash
   python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 1 --chunk-size 6000 --seed 42
   ```

---

## 🆘 Suporte

Se encontrar problemas:

1. Execute `.\server_config.ps1` para verificar configuração
2. Teste a conexão SSH manualmente: `ssh seu-servidor.com "echo OK"`
3. Verifique os logs do script para identificar o erro
4. Certifique-se de que o túnel Cloudflare está ativo

---

**Versão:** 1.0
**Data:** 2025-10-15
**Compatibilidade:** Windows PowerShell 5.1+
