"""Install experiment packages and pinned public sources inside Colab."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def checkout(name, spec, check_only):
    directory = ROOT / 'external' / name
    if not directory.exists() and not check_only:
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(directory)], check=True)
        subprocess.run(['git', '-C', str(directory), 'remote', 'add', 'origin', spec['url']], check=True)
        subprocess.run(['git', '-C', str(directory), 'fetch', '--depth', '1', 'origin', spec['revision']], check=True)
        subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != spec['revision']:
        raise RuntimeError(f'{name}: revision mismatch')
    expected = (ROOT / 'patches/cartridges.patch').read_bytes() if name == 'cartridges' else b''
    actual = subprocess.check_output(['git', '-C', str(directory), 'diff', '--binary'])
    if not actual and expected and not check_only:
        subprocess.run(['git', '-C', str(directory), 'apply', str(ROOT / 'patches/cartridges.patch')], check=True)
        actual = subprocess.check_output(['git', '-C', str(directory), 'diff', '--binary'])
    if actual != expected:
        raise RuntimeError(f'{name}: unexpected source changes')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required')
    cfg = json.loads((ROOT / 'configs/experiment.json').read_text())
    if not args.check_only:
        if sys.platform != 'linux':
            raise RuntimeError('Run setup in the Colab runtime')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--require-hashes', '-r',
                        str(ROOT / 'requirements.lock')], check=True)
    source = checkout('cartridges', cfg['upstream'], args.check_only)
    from packaging.requirements import Requirement
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        actual = importlib.metadata.version(requirement.name).split('+')[0]
        if actual not in requirement.specifier:
            raise RuntimeError(f'{requirement}: installed {actual}')
    os.environ['CARTRIDGES_DIR'] = str(source)
    os.environ['CARTRIDGES_OUTPUT_DIR'] = str(ROOT / 'artifacts')
    sys.path.insert(0, str(ROOT))
    import experiment
    for benchmark in cfg['benchmarks']:
        experiment.reference_configs(benchmark)
    import torch
    report = {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'upstream_revision': cfg['upstream']['revision'],
              'patch_sha256': experiment.sha(ROOT / 'patches/cartridges.patch'),
              'lock_sha256': experiment.sha(ROOT / 'requirements.lock'),
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    label = 'gpu' if torch.cuda.is_available() else 'local'
    experiment.write_json(ROOT / f'results/environment-{label}.json', report)
    print(json.dumps({'status': 'passed', 'environment': label, 'torch': torch.__version__, 'recipes': list(cfg['benchmarks'])}))


if __name__ == '__main__':
    main()
