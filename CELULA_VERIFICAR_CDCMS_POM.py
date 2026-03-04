# =============================================================================
# VERIFICAR: pom.xml do CDCMS.CIL para identificar versao MOA
# =============================================================================
# Este script baixa e analisa o pom.xml do CDCMS.CIL para descobrir
# contra qual versao do MOA ele foi compilado
# =============================================================================

import subprocess
import urllib.request
from pathlib import Path
import re

print("="*70)
print("VERIFICAR: Versao MOA do CDCMS.CIL")
print("="*70)

# =============================================================================
# PASSO 1: Baixar pom.xml do repositorio
# =============================================================================
print("\n--- PASSO 1: Baixar pom.xml ---")

CDCMS_POM_URL = "https://raw.githubusercontent.com/michaelchiucw/CDCMS.CIL/main/pom.xml"
CDCMS_POM_URL_ALT = "https://raw.githubusercontent.com/michaelchiucw/CDCMS.CIL/master/pom.xml"

pom_content = None

for url in [CDCMS_POM_URL, CDCMS_POM_URL_ALT]:
    try:
        print(f"Tentando: {url}")
        with urllib.request.urlopen(url, timeout=30) as response:
            pom_content = response.read().decode('utf-8')
            print(f"[OK] pom.xml baixado ({len(pom_content)} bytes)")
            break
    except Exception as e:
        print(f"  Erro: {e}")

# =============================================================================
# PASSO 2: Analisar dependencias
# =============================================================================
print("\n--- PASSO 2: Analisar dependencias ---")

if pom_content:
    print("\nConteudo do pom.xml:\n")
    print(pom_content[:2000])
    if len(pom_content) > 2000:
        print("...")

    # Procurar versao do MOA
    moa_version_match = re.search(r'<artifactId>moa</artifactId>\s*<version>([^<]+)</version>', pom_content)
    if moa_version_match:
        moa_version = moa_version_match.group(1)
        print(f"\n*** Versao MOA encontrada: {moa_version} ***")
    else:
        print("\n[AVISO] Versao MOA nao encontrada no pom.xml")

    # Procurar versao do Weka
    weka_version_match = re.search(r'<artifactId>weka[^<]*</artifactId>\s*<version>([^<]+)</version>', pom_content)
    if weka_version_match:
        weka_version = weka_version_match.group(1)
        print(f"*** Versao Weka encontrada: {weka_version} ***")
else:
    print("[ERRO] Nao foi possivel baixar pom.xml")
    print("\nTentando clonar repositorio...")

    # Tentar clonar
    clone_dir = Path('/content/cdcms_check')
    clone_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/michaelchiucw/CDCMS.CIL.git"],
        cwd=str(clone_dir),
        capture_output=True, text=True, timeout=120
    )

    pom_path = clone_dir / 'CDCMS.CIL' / 'pom.xml'
    if pom_path.exists():
        with open(pom_path) as f:
            pom_content = f.read()
        print(f"[OK] pom.xml encontrado no clone")

        # Analisar
        moa_version_match = re.search(r'<artifactId>moa</artifactId>\s*<version>([^<]+)</version>', pom_content)
        if moa_version_match:
            print(f"\n*** Versao MOA: {moa_version_match.group(1)} ***")
    else:
        # Listar arquivos no repo
        cdcms_dir = clone_dir / 'CDCMS.CIL'
        if cdcms_dir.exists():
            print("\nArquivos no repositorio:")
            for f in cdcms_dir.iterdir():
                print(f"  {f.name}")

# =============================================================================
# PASSO 3: Verificar versao MOA no MOA-dependencies.jar
# =============================================================================
print("\n--- PASSO 3: Versao MOA no MOA-dependencies.jar ---")

MOA_DEPS_JAR = Path('/content/rose_jars/MOA-dependencies.jar')

if MOA_DEPS_JAR.exists():
    # Extrair MANIFEST.MF
    result = subprocess.run(
        f'unzip -p "{MOA_DEPS_JAR}" META-INF/MANIFEST.MF 2>/dev/null',
        shell=True, capture_output=True, text=True
    )

    if result.stdout:
        print("MANIFEST.MF do MOA-dependencies.jar:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")

    # Tentar encontrar versao no nome de pacote
    result2 = subprocess.run(
        f'jar tf "{MOA_DEPS_JAR}" 2>/dev/null | grep "moa/.*\\.class$" | head -1',
        shell=True, capture_output=True, text=True
    )

    # Procurar por arquivo de versao
    result3 = subprocess.run(
        f'unzip -p "{MOA_DEPS_JAR}" "*/version.txt" 2>/dev/null',
        shell=True, capture_output=True, text=True
    )
    if result3.stdout:
        print(f"\nversion.txt: {result3.stdout.strip()}")
else:
    print("[AVISO] MOA-dependencies.jar nao encontrado")

# =============================================================================
# PASSO 4: Sugerir solucao baseada nas versoes
# =============================================================================
print("\n--- PASSO 4: Recomendacoes ---")

print("""
ANALISE:

O problema fundamental e que o CDCMS.CIL JAR foi compilado contra uma versao
especifica do MOA, e o MOA-dependencies.jar do ROSE contem outra versao.

SOLUCOES POSSIVEIS:

1. BAIXAR MOA CORRETO:
   Se o pom.xml do CDCMS.CIL especifica MOA 2020.07.1:
   - Baixar de: https://mvnrepository.com/artifact/nz.ac.waikato.cms.moa/moa/2020.07.1

2. RECOMPILAR CDCMS.CIL:
   - Clonar repositorio: git clone https://github.com/michaelchiucw/CDCMS.CIL
   - Compilar com Maven: mvn clean package -DskipTests
   - Usar JAR resultante (target/cdcms-*-jar-with-dependencies.jar)

3. CONTATAR AUTORES:
   - Email/Issue no GitHub pedindo JAR funcional
   - Perguntar qual ambiente eles usam para executar

4. VERIFICAR RELEASES:
   - Ver se ha releases pre-compiladas no GitHub
   - URL: https://github.com/michaelchiucw/CDCMS.CIL/releases
""")

# =============================================================================
# PASSO 5: Tentar baixar MOA especifico
# =============================================================================
print("\n--- PASSO 5: Opcao - Baixar MOA do Maven ---")

# Se encontramos versao no pom.xml, tentar baixar
moa_version = None
if pom_content:
    match = re.search(r'<artifactId>moa</artifactId>\s*<version>([^<]+)</version>', pom_content)
    if match:
        moa_version = match.group(1)

if moa_version:
    print(f"\nVersao MOA necessaria: {moa_version}")

    maven_url = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/{moa_version}/moa-{moa_version}.jar"
    print(f"URL Maven: {maven_url}")

    moa_specific_jar = Path(f'/content/moa_jars/moa-{moa_version}.jar')
    moa_specific_jar.parent.mkdir(exist_ok=True)

    print(f"\nBaixando moa-{moa_version}.jar...")
    try:
        urllib.request.urlretrieve(maven_url, moa_specific_jar)
        if moa_specific_jar.exists():
            print(f"[OK] Baixado: {moa_specific_jar.stat().st_size/(1024*1024):.2f} MB")
        else:
            print("[FALHA] Download falhou")
    except Exception as e:
        print(f"[ERRO] {e}")

    # Se baixou, tentar executar
    if moa_specific_jar.exists():
        print(f"\nTestando com moa-{moa_version}.jar...")

        # Testar com cdcms_cil_minimal.jar ou cdcms_cil_clean.jar
        cdcms_jars = [
            Path('/content/cdcms_jars/cdcms_cil_minimal.jar'),
            Path('/content/cdcms_jars/cdcms_cil_clean.jar'),
            Path('/content/cdcms_jars/cdcms_cil.jar')
        ]

        cdcms_jar = None
        for j in cdcms_jars:
            if j.exists():
                cdcms_jar = j
                break

        if cdcms_jar:
            classpath = f"{cdcms_jar}:{moa_specific_jar}"
            print(f"Classpath: {cdcms_jar.name}:{moa_specific_jar.name}")

            # Teste simples
            test_cmd = [
                "java", "-Xmx2g",
                "-cp", classpath,
                "moa.DoTask",
                "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
            ]

            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr

            if "Exception" not in output and result.returncode == 0:
                print("[SUCESSO] CDCMS.CIL reconhecido!")
                for line in output.strip().split('\n')[:5]:
                    print(f"  {line[:70]}")
            else:
                print("[FALHA]")
                for line in output.strip().split('\n')[:3]:
                    print(f"  {line[:70]}")

                # Pode precisar de mais dependencias
                print("\n[NOTA] MOA standalone pode precisar de dependencias Weka")
                print("Tentando baixar dependencias...")

                # Weka
                weka_version = "3.9.4"  # versao comum
                weka_url = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/{weka_version}/weka-stable-{weka_version}.jar"
                weka_jar = Path(f'/content/moa_jars/weka-stable-{weka_version}.jar')

                try:
                    print(f"Baixando weka-stable-{weka_version}.jar...")
                    urllib.request.urlretrieve(weka_url, weka_jar)
                    if weka_jar.exists():
                        print(f"[OK] {weka_jar.stat().st_size/(1024*1024):.1f} MB")

                        # Testar com Weka
                        classpath2 = f"{cdcms_jar}:{moa_specific_jar}:{weka_jar}"
                        test_cmd2 = [
                            "java", "-Xmx2g",
                            "-cp", classpath2,
                            "moa.DoTask",
                            "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
                        ]

                        result2 = subprocess.run(test_cmd2, capture_output=True, text=True, timeout=30)
                        output2 = result2.stdout + result2.stderr

                        if "Exception" not in output2:
                            print("\n[SUCESSO com Weka] CDCMS.CIL reconhecido!")
                        else:
                            print("\n[FALHA com Weka]")
                            for line in output2.strip().split('\n')[:3]:
                                print(f"  {line[:70]}")
                except Exception as e:
                    print(f"[ERRO Weka] {e}")
        else:
            print("[AVISO] Nenhum JAR CDCMS encontrado")
else:
    print("[SKIP] Versao MOA nao identificada")

print("\n" + "="*70)
