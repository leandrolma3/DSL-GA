# ============================================================================
# Configuração de Servidor SSH
# Este arquivo permite alterar o servidor sem modificar o script principal
# ============================================================================

# Arquivo de configuração do servidor
$CONFIG_FILE = "server_address.txt"

function Show-Menu {
    Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║        Configuração de Servidor SSH                   ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Ver servidor atual" -ForegroundColor White
    Write-Host "2. Alterar servidor" -ForegroundColor White
    Write-Host "3. Testar conexão" -ForegroundColor White
    Write-Host "4. Restaurar servidor padrão" -ForegroundColor White
    Write-Host "5. Sair" -ForegroundColor White
    Write-Host ""
}

function Get-CurrentServer {
    if (Test-Path $CONFIG_FILE) {
        return Get-Content $CONFIG_FILE -Raw
    } else {
        return "ssh right-scotia-humor-toward.trycloudflare.com"
    }
}

function Set-Server {
    param([string]$ServerAddress)
    $ServerAddress | Out-File -FilePath $CONFIG_FILE -Encoding UTF8 -NoNewline
    Write-Host "  ✓ Servidor configurado: $ServerAddress" -ForegroundColor Green
}

function Test-ServerConnection {
    param([string]$ServerAddress)

    Write-Host "`n  Testando conexão com: $ServerAddress" -ForegroundColor Cyan
    $result = ssh $ServerAddress 'echo "OK"' 2>&1

    if ($LASTEXITCODE -eq 0 -and $result -match "OK") {
        Write-Host "  ✓ Conexão estabelecida com sucesso!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  ✗ Falha na conexão" -ForegroundColor Red
        Write-Host "  Erro: $result" -ForegroundColor Yellow
        return $false
    }
}

# Loop principal
do {
    Show-Menu
    $choice = Read-Host "Escolha uma opção"

    switch ($choice) {
        "1" {
            $currentServer = Get-CurrentServer
            Write-Host "`n  Servidor atual: $currentServer" -ForegroundColor Cyan
        }
        "2" {
            Write-Host "`n  Digite o novo endereço do servidor:" -ForegroundColor Cyan
            Write-Host "  (Exemplo: ssh user@hostname.com ou ssh tunnel.trycloudflare.com)" -ForegroundColor Gray
            $newServer = Read-Host "  Servidor"

            if ($newServer -ne "") {
                Set-Server -ServerAddress $newServer
            } else {
                Write-Host "  ✗ Endereço inválido" -ForegroundColor Red
            }
        }
        "3" {
            $currentServer = Get-CurrentServer
            Test-ServerConnection -ServerAddress $currentServer
        }
        "4" {
            $defaultServer = "ssh right-scotia-humor-toward.trycloudflare.com"
            Set-Server -ServerAddress $defaultServer
        }
        "5" {
            Write-Host "`n  Até logo!" -ForegroundColor Cyan
            break
        }
        default {
            Write-Host "`n  ✗ Opção inválida" -ForegroundColor Red
        }
    }

    if ($choice -ne "5") {
        Write-Host "`n  Pressione qualquer tecla para continuar..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }

} while ($choice -ne "5")
