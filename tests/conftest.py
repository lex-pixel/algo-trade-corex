"""
tests/conftest.py
==================
pytest global konfigürasyonu.

- Test sirasinda loguru FILE handler'larini devre disi birakir.
  Boylece tests/ altindaki hicbir test, gercek logs/ dosyalarina
  (errors.log, algo_trade.log, trades.log) yazamaz.
- Sadece stdout'a (terminal) yazilir — log izolasyonu saglanir.
"""

import pytest
from loguru import logger


def pytest_configure(config):
    """
    pytest baslarken cagrilir (herhangi bir test toplanmadan once).
    Tum loguru sink'lerini kaldir, sadece stderr'e yaz.
    """
    logger.remove()   # Tum mevcut handler'lari sil (dosya dahil)
    # Testlerde sadece WARNING+ seviyesini terminale yaz (gurultu azalt)
    logger.add(
        __import__("sys").stderr,
        level="WARNING",
        format="<level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
        colorize=False,
    )
