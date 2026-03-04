#!/usr/bin/env python3
"""
Setup do ERulesD2S

Clona repositorio, compila e prepara ambiente para execucao.
"""

import subprocess
import sys
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_header(message):
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80 + "\n")


def check_command(cmd):
    """Verifica se comando esta disponivel"""
    result = subprocess.run(
        ["which", cmd] if sys.platform != "win32" else ["where", cmd],
        capture_output=True
    )
    return result.returncode == 0


def main():
    print_header("SETUP ERULESD2S")

    # 1. Verificar prerequisitos
    print_header("1. VERIFICANDO PREREQUISITOS")

    # Java
    if not check_command("java"):
        logger.error("Java nao encontrado! Instale Java 11+")
        return 1

    java_result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    logger.info(f"Java encontrado: {java_result.stderr.split()[2]}")

    # Maven
    if not check_command("mvn"):
        logger.error("Maven nao encontrado! Instale Maven")
        return 1

    mvn_result = subprocess.run(["mvn", "--version"], capture_output=True, text=True)
    logger.info(f"Maven encontrado")

    # Git
    if not check_command("git"):
        logger.error("Git nao encontrado!")
        return 1

    logger.info("Git encontrado")

    # 2. Clonar repositorio ERulesD2S
    print_header("2. CLONANDO REPOSITORIO ERULESD2S")

    erulesd2s_dir = Path("ERulesD2S")

    if erulesd2s_dir.exists():
        logger.info(f"Diretorio {erulesd2s_dir} ja existe")
        response = input("Deseja remover e reclonar? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(erulesd2s_dir)
        else:
            logger.info("Usando diretorio existente")
            return 0

    logger.info("Clonando repositorio...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/canoalberto/ERulesD2S.git"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Falha ao clonar: {result.stderr}")
        return 1

    logger.info("Repositorio clonado com sucesso")

    # 3. Compilar projeto
    print_header("3. COMPILANDO ERULESD2S")

    logger.info("Compilando com Maven... (pode demorar alguns minutos)")

    # Navegar para diretorio
    original_dir = Path.cwd()

    try:
        import os
        os.chdir(erulesd2s_dir)

        # Clean e compile
        logger.info("Executando: mvn clean compile")
        result = subprocess.run(
            ["mvn", "clean", "compile", "-DskipTests"],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )

        if result.returncode != 0:
            logger.error(f"Compilacao falhou: {result.stderr[-500:]}")
            return 1

        logger.info("Compilacao concluida")

        # Package
        logger.info("Executando: mvn package")
        result = subprocess.run(
            ["mvn", "package", "-DskipTests"],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            logger.error(f"Package falhou: {result.stderr[-500:]}")
            return 1

        logger.info("Package concluido")

    finally:
        os.chdir(original_dir)

    # 4. Verificar JARs
    print_header("4. VERIFICANDO JARS")

    jar_dir = erulesd2s_dir / "target"
    jars = list(jar_dir.glob("*.jar"))

    if not jars:
        logger.error("Nenhum JAR encontrado em target/")
        return 1

    logger.info(f"JARs encontrados:")
    for jar in jars:
        size_mb = jar.stat().st_size / (1024 * 1024)
        logger.info(f"  - {jar.name} ({size_mb:.2f} MB)")

    # 5. Criar link simbolico para facilitar acesso
    print_header("5. CONFIGURANDO AMBIENTE")

    # Encontrar JAR principal
    main_jar = None
    for jar in jars:
        if "with-dependencies" in jar.name or jar.stat().st_size > 1024 * 1024:
            main_jar = jar
            break

    if not main_jar:
        main_jar = jars[0]

    logger.info(f"JAR principal: {main_jar.name}")

    # Criar link
    link_path = Path("erulesd2s.jar")
    if link_path.exists():
        link_path.unlink()

    try:
        link_path.symlink_to(main_jar.resolve())
        logger.info(f"Link criado: erulesd2s.jar -> {main_jar}")
    except:
        # Fallback: copiar arquivo
        shutil.copy(main_jar, link_path)
        logger.info(f"JAR copiado para: erulesd2s.jar")

    # 6. Teste rapido
    print_header("6. TESTE RAPIDO")

    logger.info("Testando execucao do JAR...")
    result = subprocess.run(
        ["java", "-jar", str(link_path)],
        capture_output=True,
        text=True,
        timeout=10
    )

    # Esperamos que falhe com mensagem de uso
    if "moa" in result.stderr.lower() or "erulesd2s" in result.stderr.lower():
        logger.info("JAR executavel confirmado")
    else:
        logger.warning("Nao foi possivel confirmar funcionalidade do JAR")

    # 7. Criar arquivo de configuracao
    print_header("7. CRIANDO ARQUIVO DE CONFIGURACAO")

    config_content = f"""# Configuracao ERulesD2S
ERULESD2S_JAR={link_path.absolute()}
ERULESD2S_DIR={erulesd2s_dir.absolute()}
JAVA_MEMORY=8g
"""

    config_file = Path("erulesd2s_config.env")
    with open(config_file, 'w') as f:
        f.write(config_content)

    logger.info(f"Configuracao salva em: {config_file}")

    # Conclusao
    print_header("SETUP CONCLUIDO")
    print("ERulesD2S instalado e configurado com sucesso!")
    print()
    print("Arquivos importantes:")
    print(f"  - JAR: {link_path}")
    print(f"  - Codigo fonte: {erulesd2s_dir}")
    print(f"  - Config: {config_file}")
    print()
    print("Proximo passo:")
    print("  python test_erulesd2s_integration.py")
    print()

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
