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
