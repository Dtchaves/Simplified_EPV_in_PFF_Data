#!/usr/bin/env python3
"""Run actual notebook/helper fixtures and save a hash-bound gate artifact."""
from pathlib import Path
import argparse
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

from PublicCausal.notebook_runtime import (
    export_notebook_core, file_hash, atomic_write_json, environment_provenance,
    RunLogger,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Isolated source-bound manifest destination')
    args = parser.parse_args(argv)
    target = (args.output_dir or ROOT / 'data/public_statsbomb_causal/manifests').resolve()
    logger = RunLogger(target / 'validation', 'implementation-validation')
    try:
        notebook = ROOT / 'results/models/public_statsbomb_causal_study.ipynb'
        _, core_hash = export_notebook_core(notebook, target / 'validation/runtime')
        logger.event('tests_start', stage='fixtures', core_source_hash=core_hash)
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/public_causal'), pattern='test_*.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        record = dict(passed=result.testsRun > 0 and result.wasSuccessful() and not result.skipped,
                      validation_error='No implementation tests discovered' if result.testsRun == 0 else None,
                      tests_run=result.testsRun, skipped=result.skipped,
                      failures=[(str(test), message) for test, message in result.failures],
                      errors=[(str(test), message) for test, message in result.errors],
                      core_source_hash=core_hash,
                      helper_hashes={str(path.relative_to(ROOT)): file_hash(path) for path in (ROOT / 'src/PublicCausal').glob('*.py')},
                      environment=environment_provenance())
        atomic_write_json(target / 'implementation_tests.json', record)
        logger.close('completed' if record['passed'] else 'failed', tests_run=result.testsRun,
                     failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped))
        return 0 if record['passed'] else 1
    except BaseException:
        logger.close('failed')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
