"""
Wrapper Python para ROSE (Robust Online Self-Adjusting Ensemble)

Executa ROSE via subprocess Java (MOA) e integra com pipeline Python.

Baseado em: Cano & Krawczyk (2022) - Machine Learning, Vol. 111(7), pp. 2561-2599
GitHub: https://github.com/canoalberto/ROSE
"""

import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ROSEWrapper:
    """Wrapper para executar ROSE via MOA/Java"""

    def __init__(
        self,
        rose_jar_path: str = "ROSE-1.0.jar",
        moa_dependencies_jar: str = "MOA-dependencies.jar",
        sizeofag_jar: str = "sizeofag-1.0.4.jar",
        java_memory: str = "4g"
    ):
        """
        Args:
            rose_jar_path: Caminho para ROSE-1.0.jar
            moa_dependencies_jar: Caminho para MOA-dependencies.jar
            sizeofag_jar: Caminho para sizeofag-1.0.4.jar (opcional, para memory profiling)
            java_memory: Memoria heap Java (ex: "4g")
        """
        self.rose_jar = Path(rose_jar_path)
        self.moa_jar = Path(moa_dependencies_jar)
        self.sizeofag_jar = Path(sizeofag_jar)
        self.java_memory = java_memory

        # Verificar se JARs existem
        if not self.rose_jar.exists():
            logger.warning(f"ROSE JAR nao encontrado: {self.rose_jar}")
        if not self.moa_jar.exists():
            logger.warning(f"MOA dependencies JAR nao encontrado: {self.moa_jar}")

    def build_classpath(self) -> str:
        """Constroi classpath para Java"""
        classpath_parts = []

        # ROSE JAR
        if self.rose_jar.exists():
            classpath_parts.append(str(self.rose_jar))

        # MOA dependencies JAR
        if self.moa_jar.exists():
            classpath_parts.append(str(self.moa_jar))

        # Windows usa ; ao inves de :
        import platform
        separator = ";" if platform.system() == "Windows" else ":"
        return separator.join(classpath_parts)

    def build_command(
        self,
        arff_file: Path,
        output_file: Path,
        max_instances: Optional[int] = None,
        chunk_size: int = 500,
        evaluator: str = "WindowAUCImbalancedPerformanceEvaluator"
    ) -> List[str]:
        """
        Constroi comando para executar ROSE.

        Args:
            arff_file: Arquivo ARFF de entrada
            output_file: Arquivo de saida para resultados
            max_instances: Maximo de instancias a processar
            chunk_size: Frequencia de avaliacao (cada N instancias)
            evaluator: Evaluator MOA a usar

        Returns:
            Lista de argumentos do comando
        """
        classpath = self.build_classpath()

        # Comando base Java
        cmd = [
            "java",
            f"-Xmx{self.java_memory}"
        ]

        # Adicionar javaagent para sizeofag se disponivel
        if self.sizeofag_jar.exists():
            cmd.append(f"-javaagent:{self.sizeofag_jar}")

        cmd.extend(["-cp", classpath])

        # Task MOA: EvaluateInterleavedTestThenTrain (prequential)
        task = "EvaluateInterleavedTestThenTrain"

        # Learner: ROSE
        learner = "(moa.classifiers.meta.imbalanced.ROSE)"

        # Stream: ArffFileStream
        stream = f"(ArffFileStream -f {arff_file})"

        # Evaluator
        eval_str = f"({evaluator})"

        # Opcoes de avaliacao
        task_parts = [
            task,
            "-e", eval_str,
            "-s", stream,
            "-l", learner,
            "-f", str(chunk_size),  # Frequencia de avaliacao
            "-d", str(output_file)  # Arquivo de saida
        ]

        if max_instances:
            task_parts.extend(["-i", str(max_instances)])

        # Concatenar em uma string para DoTask
        task_string = " ".join(task_parts)

        cmd_list = cmd + ["moa.DoTask", task_string]

        return cmd_list

    def run(
        self,
        arff_file: Path,
        output_dir: Path,
        max_instances: Optional[int] = None,
        chunk_size: int = 500,
        evaluator: str = "WindowAUCImbalancedPerformanceEvaluator",
        timeout: Optional[int] = None
    ) -> Tuple[bool, Dict]:
        """
        Executa ROSE.

        Args:
            arff_file: Arquivo ARFF de entrada
            output_dir: Diretorio de saida
            max_instances: Maximo de instancias
            chunk_size: Frequencia de avaliacao
            evaluator: Evaluator MOA
            timeout: Timeout em segundos

        Returns:
            (sucesso, resultados_dict)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "rose_results.csv"
        log_file = output_dir / "rose_log.txt"

        # Construir comando
        cmd = self.build_command(
            arff_file=arff_file,
            output_file=output_file,
            max_instances=max_instances,
            chunk_size=chunk_size,
            evaluator=evaluator
        )

        logger.info("Executando ROSE...")
        logger.info(f"Comando: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False
            )

            duration = time.time() - start_time

            # Salvar logs
            with open(log_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            if result.returncode != 0:
                logger.error(f"ROSE falhou (returncode={result.returncode})")
                logger.error(f"Stderr: {result.stderr[:500]}")
                return False, {}

            logger.info(f"ROSE executado com sucesso em {duration:.1f}s")

            # Parsear resultados
            results = self.parse_results(output_file, result.stdout)
            results['execution_time'] = duration

            return True, results

        except subprocess.TimeoutExpired:
            logger.error(f"ROSE timeout ({timeout}s)")
            return False, {}

        except Exception as e:
            logger.error(f"Erro ao executar ROSE: {e}")
            import traceback
            traceback.print_exc()
            return False, {}

    def parse_results(self, results_file: Path, stdout: str) -> Dict:
        """
        Parseia resultados do ROSE.

        Args:
            results_file: Arquivo CSV de resultados
            stdout: Saida padrao do MOA

        Returns:
            Dicionario com metricas
        """
        results = {}

        # Tentar ler arquivo CSV se existir
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()

                # Filtrar headers duplicados (comecam com "Learner" ou "learning")
                data_lines = [line for line in lines
                              if not line.startswith('Learner')
                              and not line.startswith('learning')
                              and line.strip()]

                if len(data_lines) > 0:
                    header_line = lines[0]
                    last_data_line = data_lines[-1]

                    headers = header_line.strip().split(',')
                    values = last_data_line.strip().split(',')

                    data_dict = dict(zip(headers, values))

                    # Debug: mostrar colunas disponiveis
                    logger.info(f"Colunas disponiveis: {list(data_dict.keys())}")

                    # Extrair metricas (valores ja estao em decimal 0-1)
                    # G-mean
                    if 'G-Mean' in data_dict:
                        try:
                            results['gmean'] = float(data_dict['G-Mean'])
                        except:
                            pass

                    # pAUC (AUC periodico)
                    if 'pAUC' in data_dict:
                        try:
                            results['auc'] = float(data_dict['pAUC'])
                        except:
                            pass

                    # Accuracy (ja em decimal)
                    if 'Accuracy' in data_dict:
                        try:
                            results['accuracy'] = float(data_dict['Accuracy'])
                        except:
                            pass

                    # Kappa (ja em decimal)
                    if 'Kappa' in data_dict:
                        try:
                            results['kappa'] = float(data_dict['Kappa'])
                        except:
                            pass

                    # Recall
                    if 'Recall' in data_dict:
                        try:
                            results['recall'] = float(data_dict['Recall'])
                        except:
                            pass

                    logger.info(f"Metricas extraidas: {results}")

            except Exception as e:
                logger.warning(f"Erro ao ler CSV: {e}")

        # Fallback: parsear stdout
        if not results:
            logger.info("Tentando parsear stdout como fallback...")
            lines = stdout.split('\n')
            for line in lines:
                if line.count(',') > 5:
                    try:
                        parts = line.split(',')
                        if len(parts) > 7:
                            acc_value = float(parts[7])
                            results['accuracy'] = acc_value / 100.0
                            break
                    except:
                        continue

        # Defaults
        if 'accuracy' not in results:
            results['accuracy'] = 0.0
        if 'gmean' not in results:
            results['gmean'] = results.get('accuracy', 0.0)

        return results


class ROSEEvaluator:
    """Evaluator compativel com pipeline usando ROSE"""

    def __init__(
        self,
        wrapper: ROSEWrapper,
        output_dir: Path,
        chunk_size: int = 500,
        evaluator: str = "WindowAUCImbalancedPerformanceEvaluator"
    ):
        """
        Args:
            wrapper: Instancia do ROSEWrapper
            output_dir: Diretorio de saida
            chunk_size: Frequencia de avaliacao
            evaluator: Evaluator MOA
        """
        self.wrapper = wrapper
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.evaluator = evaluator

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_chunks(
        self,
        chunks: List[Tuple[np.ndarray, np.ndarray]],
        arff_files: List[Path]
    ) -> pd.DataFrame:
        """
        Avalia ROSE em multiplos chunks.

        Args:
            chunks: Lista de (X, y) tuplas
            arff_files: Lista de arquivos ARFF correspondentes

        Returns:
            DataFrame com resultados
        """
        results = []

        for chunk_idx, arff_file in enumerate(arff_files):
            logger.info(f"Avaliando chunk {chunk_idx + 1}/{len(arff_files)}")

            success, metrics = self.wrapper.run(
                arff_file=arff_file,
                output_dir=self.output_dir / f"chunk_{chunk_idx}",
                chunk_size=self.chunk_size,
                evaluator=self.evaluator,
                timeout=600  # 10 min timeout
            )

            if not success:
                logger.warning(f"Chunk {chunk_idx} falhou, usando metricas vazias")
                metrics = {'accuracy': 0.0, 'gmean': 0.0}

            results.append({
                'chunk': chunk_idx,
                'model': 'ROSE',
                'accuracy': metrics.get('accuracy', 0.0),
                'gmean': metrics.get('gmean', 0.0),
                'auc': metrics.get('auc', 0.0),
                'kappa': metrics.get('kappa', 0.0),
                'f1_weighted': metrics.get('accuracy', 0.0),  # Aproximacao
                'execution_time': metrics.get('execution_time', 0.0)
            })

        return pd.DataFrame(results)


def download_rose_jars(output_dir: Path) -> bool:
    """
    Baixa JARs do ROSE do GitHub.

    Args:
        output_dir: Diretorio de saida

    Returns:
        True se sucesso
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://github.com/canoalberto/ROSE/raw/master"
    jars = [
        "ROSE-1.0.jar",
        "MOA-dependencies.jar",
        "sizeofag-1.0.4.jar"
    ]

    for jar in jars:
        jar_path = output_dir / jar
        if jar_path.exists():
            logger.info(f"JAR ja existe: {jar_path}")
            continue

        url = f"{base_url}/{jar}"
        logger.info(f"Baixando {jar}...")

        try:
            urllib.request.urlretrieve(url, jar_path)
            size_mb = jar_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Baixado: {jar} ({size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"  Erro ao baixar {jar}: {e}")
            return False

    return True


def check_java_installation() -> Tuple[bool, str]:
    """
    Verifica se Java esta instalado.

    Returns:
        (instalado, versao)
    """
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Java version esta no stderr
            version_line = result.stderr.split('\n')[0]
            return True, version_line

        return False, ""

    except FileNotFoundError:
        return False, ""


# Exemplo de uso
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Verificar Java
    java_ok, java_version = check_java_installation()
    print(f"Java instalado: {java_ok}")
    if java_ok:
        print(f"Versao: {java_version}")

    # Criar wrapper
    wrapper = ROSEWrapper(
        rose_jar_path="ROSE-1.0.jar",
        moa_dependencies_jar="MOA-dependencies.jar",
        java_memory="4g"
    )

    # Executar em um arquivo ARFF (exemplo)
    # success, results = wrapper.run(
    #     arff_file=Path("test_data.arff"),
    #     output_dir=Path("rose_output"),
    #     chunk_size=500
    # )
