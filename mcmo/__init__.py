"""
MCMO - Multistream Classification based on Multi-objective Optimization

Reduced-space Multistream Classification Framework based on Multi-objective Optimization
Paper: IEEE TEVC 2023
GitHub: https://github.com/Jesen-BT/MCMO
"""

# Try to import MCMO components, but don't fail if dependencies missing
try:
    from .MCMO import MCMO
    from .GMM import DGMM
    MCMO_AVAILABLE = True
    __all__ = ['MCMO', 'DGMM', 'MCMO_AVAILABLE']
except ImportError as e:
    MCMO_AVAILABLE = False
    IMPORT_ERROR = str(e)
    __all__ = ['MCMO_AVAILABLE', 'IMPORT_ERROR']
    # Note: baseline_mcmo can still be imported as it has its own try/except
