#!/usr/bin/env python3
"""Check default binary headers, identifiers, metadata and publication."""
import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


def header(data, magic):
    assert data[:8] == magic
    version, size = struct.unpack_from('<II', data, 8)
    assert version == 1 and size <= 1024 * 1024
    return json.loads(data[16:16 + size]), 16 + size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='dacapo-binary-') as directory:
        temp = Path(directory)
        prefix = temp / 'binary'
        options = (f'prefix={prefix} target-id=test operator-spec-id=test '
                   f'operator-spec-sha256=sha256:{"0" * 64} context-id=test report-io=true')
        command = [str(args.hecate_opt), str(args.source_dir / 'test/runtime-plan/bundle-reuse.mlir'),
                   f'-p=builtin.module(func.func(convert-earth-to-ckks,emit-runtime-plan{{{options}}}))',
                   '-o', '/dev/null']
        emitted = subprocess.run(command, cwd=args.source_dir, check=True, capture_output=True, text=True)
        plan = Path(f'{prefix}.bundle_reuse.runtime-plan.bin')
        data = plan.read_bytes()
        metadata, cursor = header(data, b'CKKSPL01')
        assert metadata['plaintext_bundle']['manifest_format'] == 'binary'
        assert 'manifest_sha256' not in metadata['plaintext_bundle']
        assert all(metadata[name] == [] for name in ('values', 'external_inputs', 'initialization', 'execution', 'finalization', 'final_outputs'))
        report = json.loads(next(line[len('runtime-plan-io '):] for line in emitted.stderr.splitlines()
                                 if line.startswith('runtime-plan-io ')))
        assert report['plan_format'] == 'binary' and report['plan_hash_seconds'] == 0
        assert report['plan_sha256'] == '' and report['plan_bytes'] == len(data)
        bundle = Path(f'{prefix}.bundle_reuse.bundle')
        manifest = (bundle / 'manifest.bin').read_bytes()
        manifest_metadata, cursor = header(manifest, b'CKKSMF01')
        assert manifest_metadata['bundle_id'] == metadata['plaintext_bundle']['id']
        assert manifest_metadata['pack_byte_length'] == (bundle / 'data.bin').stat().st_size == 4800
        count, = struct.unpack_from('<Q', manifest, cursor)
        assert count == 1 and len(manifest) == cursor + 8 + 48
        _, offset, length = struct.unpack_from('<32sQQ', manifest, cursor + 8)
        assert offset == 0 and length == 4800
        assert not list(temp.glob('*.json')) and not list(temp.rglob('*.tmp'))
        subprocess.run(command, cwd=args.source_dir, check=True, capture_output=True, text=True)
        assert plan.read_bytes() == data
        blocked = Path(f'{prefix}.bundle_reuse.records.tmp')
        blocked.mkdir()
        failed = subprocess.run(command, cwd=args.source_dir, capture_output=True, text=True)
        assert failed.returncode != 0 and 'cannot open binary record file' in failed.stderr
        assert blocked.is_dir() and plan.read_bytes() == data
        assert not list(temp.glob('*.bundle.tmp'))
    print('Default binary export checks passed')


if __name__ == '__main__':
    main()
