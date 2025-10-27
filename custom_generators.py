# custom_generators.py (VERSÃO FINAL E CORRIGIDA)
# This file contains custom stream generators that are not available in the River library.

from river.datasets.base import SyntheticDataset
from river import base # Importando o módulo base para ter acesso ao enum Task
import numpy as np

class AssetNegotiation(SyntheticDataset):
    """
    A custom generator to simulate the AssetNegotiation dataset from the Bartosz et al. paper.
    The original paper does not specify the generation formula, so this is a plausible implementation
    based on its characteristics (5 features, configurable functions).

    The generator produces 5 numerical features, all in the range [0, 1].
    """
    def __init__(self, classification_function: int = 1, seed: int = None):
        # A CORREÇÃO FINAL ESTÁ NA LINHA 'task=' ABAIXO.
        # Usamos a string 'base.BINARY_CLF' que é a forma correta de referenciar a tarefa no River.
        super().__init__(
            n_features=5,
            n_classes=2,
            n_outputs=1,
            task=base.Classifier # <<< CORREÇÃO DEFINITIVA APLICADA AQUI
        )
        self.classification_function = classification_function
        self.seed = seed
        self._rng = np.random.RandomState(self.seed)

    def __iter__(self):
        while True:
            # Generate 5 random features between 0 and 1
            x_vals = self._rng.rand(self.n_features)
            
            x = {f'x_{i}': val for i, val in enumerate(x_vals)}
            y = 0

            # Define plausible rules for each function (F2, F3, F4)
            try:
                if self.classification_function == 1: # Corresponds to F2
                    if x['x_0'] + x['x_1'] > 1.2:
                        y = 1
                elif self.classification_function == 2: # Corresponds to F3
                    if x['x_2'] > 0.6 and x['x_3'] < 0.4:
                        y = 1
                elif self.classification_function == 3: # Corresponds to F4
                    if (x['x_4'] * 2) > (x['x_0'] + x['x_1']):
                        y = 1
                else:
                    # Default rule
                    if x['x_0'] > 0.5:
                        y = 1
            except KeyError:
                y = 0
            
            yield x, y