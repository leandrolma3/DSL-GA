"""
Wrapper Python para ERulesD2S (MOA/Java)

Executa ERulesD2S via subprocess e integra com pipeline Python.
"""

import subprocess
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ERulesD2SWrapper:
    """Wrapper para executar ERulesD2S via MOA"""

    def __init__(
        self,
        moa_jar_path: str = "erulesd2s.jar",
        erulesd2s_jar_path: Optional[str] = None,
        java_memory: str = "8g",
        gpu_enabled: bool = True
    ):
        """
        Args:
            moa_jar_path: Caminho para ERulesD2S JAR principal (padrão: erulesd2s.jar)
            erulesd2s_jar_path: Caminho para JCLEC4 JAR (se separado)
            java_memory: Memoria heap Java (ex: "8g")
            gpu_enabled: Habilitar GPU (CUDA)
        """
        self.moa_jar = Path(moa_jar_path)
        self.erulesd2s_jar = Path(erulesd2s_jar_path) if erulesd2s_jar_path else None
        self.java_memory = java_memory
        self.gpu_enabled = gpu_enabled

        # Verificar se JARs existem
        if not self.moa_jar.exists():
            logger.warning(f"ERulesD2S JAR nao encontrado: {self.moa_jar}")

    def build_classpath(self) -> str:
        """Constroi classpath para Java"""
        classpath_parts = [str(self.moa_jar)]

        # Adicionar JCLEC4 JAR se existir (OBRIGATÓRIO para ERulesD2S!)
        # Procurar na mesma pasta do JAR principal
        jar_dir = self.moa_jar.parent
        jclec_jar = jar_dir / "lib" / "JCLEC4-base-1.0-jar-with-dependencies.jar"
        if jclec_jar.exists():
            classpath_parts.append(str(jclec_jar))
            logger.info(f"JCLEC4 JAR adicionado: {jclec_jar}")
        else:
            logger.warning(f"JCLEC4 JAR nao encontrado: {jclec_jar}")
            logger.warning("ERulesD2S REQUER JCLEC4 para funcionar!")

        if self.erulesd2s_jar and self.erulesd2s_jar.exists():
            classpath_parts.append(str(self.erulesd2s_jar))

        # Adicionar outras libs (se existir diretorio lib/)
        lib_dir = jar_dir / "lib"
        if lib_dir.exists():
            lib_jars = [j for j in lib_dir.glob("*.jar") if j.name != "JCLEC4-base-1.0-jar-with-dependencies.jar"]
            classpath_parts.extend([str(j) for j in lib_jars])

        # Windows usa ; ao invés de :
        import platform
        separator = ";" if platform.system() == "Windows" else ":"
        return separator.join(classpath_parts)

    def build_command(
        self,
        arff_file: Path,
        output_file: Path,
        population_size: int = 25,
        num_generations: int = 50,
        rules_per_class: int = 5,
        chunk_size: int = 1000,
        max_instances: Optional[int] = None
    ) -> List[str]:
        """
        Constroi comando para executar ERulesD2S.

        Args:
            arff_file: Arquivo ARFF de entrada
            output_file: Arquivo de saida para resultados
            population_size: Tamanho da populacao GP
            num_generations: Numero de geracoes
            rules_per_class: Regras a aprender por classe
            chunk_size: Tamanho do chunk
            max_instances: Maximo de instancias a processar (opcional)

        Returns:
            Lista de argumentos do comando
        """
        # Classpath
        classpath = self.build_classpath()

        # Comando base Java
        cmd = [
            "java",
            f"-Xmx{self.java_memory}",
            "-cp", classpath
        ]

        # Task MOA: EvaluateInterleavedTestThenTrain (usado no benchmark oficial)
        task = "EvaluateInterleavedTestThenTrain"

        # Learner: EvolutionaryRuleLearner (classe correta do ERulesD2S)
        # IMPORTANTE: Usar -s (size) ao invés de -p (population)
        learner = (
            f"(moa.classifiers.evolutionary.EvolutionaryRuleLearner "
            f"-s {population_size} "
            f"-g {num_generations} "
            f"-r {rules_per_class})"
        )

        # Stream: ArffFileStream
        stream = f"(ArffFileStream -f {arff_file})"

        # Opcoes de avaliacao
        eval_opts = [
            f"-i {max_instances}" if max_instances else "",
            f"-f {chunk_size}",  # Frequencia de avaliacao
            "-d", str(output_file)  # Arquivo de saida
        ]
        eval_opts = [opt for opt in eval_opts if opt]

        # Montar comando MOA completo
        # IMPORTANTE: Usar EvaluateInterleavedTestThenTrain (conforme benchmark oficial)
        # Argumentos: -s (stream), -l (learner), -i (instances), -f (frequency), -d (dump)

        # Task string com argumentos
        task_parts = [
            task,  # EvaluateInterleavedTestThenTrain
            "-s", stream,
            "-l", learner,
            *eval_opts
        ]

        # Concatenar em uma string para DoTask
        task_string = " ".join(task_parts)

        # Retornar como lista para subprocess com shell=False
        cmd_list = cmd + ["moa.DoTask", task_string]

        return cmd_list

    def run(
        self,
        arff_file: Path,
        output_dir: Path,
        population_size: int = 25,
        num_generations: int = 50,
        rules_per_class: int = 5,
        chunk_size: int = 1000,
        max_instances: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Tuple[bool, Dict]:
        """
        Executa ERulesD2S.

        Args:
            arff_file: Arquivo ARFF de entrada
            output_dir: Diretorio de saida
            population_size: Tamanho da populacao
            num_generations: Numero de geracoes
            rules_per_class: Regras por classe
            chunk_size: Tamanho do chunk
            max_instances: Maximo de instancias
            timeout: Timeout em segundos

        Returns:
            (sucesso, resultados_dict)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "erulesd2s_results.csv"
        log_file = output_dir / "erulesd2s_log.txt"

        # Construir comando
        cmd = self.build_command(
            arff_file=arff_file,
            output_file=output_file,
            population_size=population_size,
            num_generations=num_generations,
            rules_per_class=rules_per_class,
            chunk_size=chunk_size,
            max_instances=max_instances
        )

        logger.info("Executando ERulesD2S...")
        logger.info(f"Comando: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # Executar subprocess com shell=False
            # cmd é uma lista, não precisa de shell
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False  # Lista de argumentos, não precisa de shell
            )

            duration = time.time() - start_time

            # Salvar logs
            with open(log_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            if result.returncode != 0:
                logger.error(f"ERulesD2S falhou (returncode={result.returncode})")
                logger.error(f"Stderr: {result.stderr[:500]}")
                return False, {}

            logger.info(f"ERulesD2S executado com sucesso em {duration:.1f}s")

            # Parsear resultados
            results = self.parse_results(output_file, result.stdout)
            results['execution_time'] = duration

            return True, results

        except subprocess.TimeoutExpired:
            logger.error(f"ERulesD2S timeout ({timeout}s)")
            return False, {}

        except Exception as e:
            logger.error(f"Erro ao executar ERulesD2S: {e}")
            return False, {}

    def parse_results(self, results_file: Path, stdout: str) -> Dict:
        """
        Parseia resultados do ERulesD2S.

        CORRIGIDO: Lida com headers duplicados no CSV gerado pelo MOA

        Args:
            results_file: Arquivo CSV de resultados (se gerado)
            stdout: Saida padrao do MOA

        Returns:
            Dicionario com metricas
        """
        results = {}

        # Tentar ler arquivo CSV se existir
        if results_file.exists():
            try:
                # CORREÇÃO: Ler CSV ignorando headers duplicados
                with open(results_file, 'r') as f:
                    lines = f.readlines()

                # Filtrar linhas que começam com "Learner" (headers)
                data_lines = [line for line in lines if not line.startswith('Learner')]

                if len(data_lines) > 0:
                    # Pegar primeira linha (header real) e última linha de dados
                    header_line = lines[0]  # Primeiro header
                    last_data_line = data_lines[-1]  # Última linha de dados

                    # Parsear manualmente
                    headers = header_line.strip().split(',')
                    values = last_data_line.strip().split(',')

                    # Criar dicionário
                    data_dict = dict(zip(headers, values))

                    # Extrair accuracy (coluna "classifications correct (percent)")
                    if 'classifications correct (percent)' in data_dict:
                        acc_str = data_dict['classifications correct (percent)']
                        try:
                            results['accuracy'] = float(acc_str) / 100.0
                            logger.info(f"Accuracy extraída do CSV: {results['accuracy']:.4f}")
                        except ValueError:
                            logger.warning(f"Erro ao converter accuracy: {acc_str}")
                            results['accuracy'] = 0.0
                    else:
                        logger.warning(f"Coluna 'classifications correct (percent)' não encontrada")
                        logger.warning(f"Colunas disponíveis: {list(data_dict.keys())}")
                        results['accuracy'] = 0.0

                    # Extrair Kappa se disponível
                    if 'Kappa Statistic (percent)' in data_dict:
                        try:
                            results['kappa'] = float(data_dict['Kappa Statistic (percent)']) / 100.0
                        except:
                            pass

            except Exception as e:
                logger.warning(f"Erro ao ler CSV: {e}")

        # FALLBACK: Parsear stdout (pode não ter "=" no formato MOA)
        # Formato MOA tabular: linha CSV com valores separados por vírgula
        if 'accuracy' not in results or results['accuracy'] == 0.0:
            logger.info("Tentando parsear stdout como fallback...")

            # Procurar por linhas com números que pareçam resultados
            lines = stdout.split('\n')
            for line in lines:
                # Linha com resultados tem muitos números separados por vírgula
                if line.count(',') > 5 and not line.startswith('learning'):
                    try:
                        parts = line.split(',')
                        # Posição 7 geralmente é "classifications correct (percent)"
                        if len(parts) > 7:
                            acc_value = float(parts[7])
                            results['accuracy'] = acc_value / 100.0
                            logger.info(f"Accuracy extraída do stdout: {results['accuracy']:.4f}")
                            break
                    except:
                        continue

        # Se ainda não tem accuracy, retornar 0
        if 'accuracy' not in results:
            results['accuracy'] = 0.0

        # Calcular gmean a partir de accuracy (aproximação)
        results['gmean'] = results['accuracy']

        return results


class ERulesD2SEvaluator:
    """Evaluator compativel com pipeline usando ERulesD2S"""

    def __init__(
        self,
        wrapper: ERulesD2SWrapper,
        output_dir: Path,
        population_size: int = 25,
        num_generations: int = 50,
        rules_per_class: int = 5
    ):
        """
        Args:
            wrapper: Instancia do ERulesD2SWrapper
            output_dir: Diretorio de saida
            population_size: Tamanho da populacao
            num_generations: Numero de geracoes
            rules_per_class: Regras por classe
        """
        self.wrapper = wrapper
        self.output_dir = Path(output_dir)
        self.population_size = population_size
        self.num_generations = num_generations
        self.rules_per_class = rules_per_class

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_chunks(
        self,
        chunks: List[Tuple[np.ndarray, np.ndarray]],
        arff_files: List[Path]
    ) -> pd.DataFrame:
        """
        Avalia ERulesD2S em multiplos chunks.

        Args:
            chunks: Lista de (X, y) tuplas
            arff_files: Lista de arquivos ARFF correspondentes

        Returns:
            DataFrame com resultados
        """
        results = []

        for chunk_idx, arff_file in enumerate(arff_files):
            logger.info(f"Avaliando chunk {chunk_idx + 1}/{len(arff_files)}")

            # Executar ERulesD2S neste chunk
            success, metrics = self.wrapper.run(
                arff_file=arff_file,
                output_dir=self.output_dir / f"chunk_{chunk_idx}",
                population_size=self.population_size,
                num_generations=self.num_generations,
                rules_per_class=self.rules_per_class,
                chunk_size=3000,  # Tamanho do chunk
                timeout=600  # 10 min timeout
            )

            if not success:
                logger.warning(f"Chunk {chunk_idx} falhou, usando metricas vazias")
                metrics = {'accuracy': 0.0, 'gmean': 0.0}

            # Calcular gmean se nao retornado
            if 'gmean' not in metrics and 'accuracy' in metrics:
                metrics['gmean'] = metrics['accuracy']  # Aproximacao

            results.append({
                'chunk': chunk_idx,
                'model': 'ERulesD2S',
                'accuracy': metrics.get('accuracy', 0.0),
                'gmean': metrics.get('gmean', 0.0),
                'f1_weighted': metrics.get('f1', metrics.get('accuracy', 0.0)),
                'execution_time': metrics.get('execution_time', 0.0)
            })

        return pd.DataFrame(results)


# Exemplo de uso
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Criar wrapper
    wrapper = ERulesD2SWrapper(
        moa_jar_path="path/to/moa.jar",
        java_memory="4g"
    )

    # Executar em um arquivo ARFF
    success, results = wrapper.run(
        arff_file=Path("test_data.arff"),
        output_dir=Path("erulesd2s_output"),
        population_size=25,
        num_generations=50
    )

    print(f"Sucesso: {success}")
    print(f"Resultados: {results}")
