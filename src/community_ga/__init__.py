"""
Community detection GA implementation accompanying
"https://onlinelibrary.wiley.com/doi/10.1155/2023/4796536".
"""

try:
    from .chromosom import Chromosom
except ImportError:  # allow running without -m
    from chromosom import Chromosom

__all__ = ["Chromosom"]
