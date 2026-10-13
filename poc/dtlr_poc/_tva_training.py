"""Lazy compatibility access to trainers now owned by TVA.

The DTLR evidence export and frozen encoders do not import TVA. This bridge is
only for historical Python training APIs and their reproducibility tests.
"""
import importlib
import os
from pathlib import Path
import sys


def load_training():
    try:
        return importlib.import_module('tva.handwriting.training')
    except ModuleNotFoundError as error:
        if error.name not in ('tva', 'tva.handwriting', 'tva.handwriting.training'):
            raise
    root = Path(os.environ.get('TVA_ROOT', str(Path(__file__).resolve().parents[3] / 'TVA')))
    if not (root / 'tva/handwriting/training.py').is_file():
        raise ImportError('Tokenizer training moved to TVA. Use its train_ed_tokenizers.ipynb; '
                          'for historical API compatibility install TVA or set TVA_ROOT.')
    sys.path.insert(0, str(root))
    return importlib.import_module('tva.handwriting.training')
