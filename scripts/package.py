"""Build a portable source archive and a SHA-256 checksum."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / 'artifacts/qwen3-teacher-scaling-source.zip'
    destination.parent.mkdir(exist_ok=True)
    files = [ROOT / p for p in ['experiment.py', 'requirements.txt', 'requirements.lock',
             'README.md', 'EXPERIMENT_PLAN.md', 'LICENSE', 'NOTICE', '.gitignore']]
    for directory in ['configs', 'scripts', 'tests', 'docs', 'notebooks', 'patches']:
        files.extend(p for p in (ROOT / directory).rglob('*') if p.suffix in ['.py', '.json', '.md', '.bib', '.ipynb', '.patch'])
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(f'{checksum}  {destination.name}\n')
    print(f'{destination}\nsha256: {checksum}')


if __name__ == '__main__':
    main()
