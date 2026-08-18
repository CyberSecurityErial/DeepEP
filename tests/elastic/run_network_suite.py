"""Run the frozen EchoP Rail/incast experiment suites in lockstep per node.

Launch this script once in every participating pod with the same arguments.
Each instance executes identical benchmark subprocesses in identical order, so
the child processes rendezvous through the existing DeepEP environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _base(args: argparse.Namespace, case: str, policy: str) -> list[str]:
    benchmark = Path(__file__).with_name('bench_network_layer.py')
    return [
        sys.executable, str(benchmark),
        '--num-processes', str(args.num_processes),
        '--num-sms', str(args.num_sms),
        '--hidden', str(args.hidden),
        '--case', case,
        '--rail-balance', policy,
        '--warmups', str(args.warmups),
        '--iterations', str(args.iterations),
        '--control-interval', str(args.control_interval),
    ]


def _rail(args: argparse.Namespace) -> list[list[str]]:
    return [
        _base(args, 'rail-ring', policy) + [
            '--totals', args.rail_totals,
            '--rail-alphas', args.rail_alphas,
            '--expert-alphas', '0',
            '--source-alphas', '0',
            '--fan-ins', '1',
            '--rail-phase', 'aligned',
        ]
        for policy in ('off', 'active')
    ]


def _incast(args: argparse.Namespace) -> list[list[str]]:
    variants = (
        ('off', []),
        ('active', []),
        ('incast', ['--incast-rail-overlap', '2.0']),
        ('incast', [
            '--incast-rail-overlap', '2.0',
            '--incast-weighted-quotas',
        ]),
    )
    return [
        _base(args, 'incast', policy) + [
            '--totals', args.incast_totals,
            '--rail-alphas', '0',
            '--expert-alphas', '0',
            '--source-alphas', '0',
            '--fan-ins', '1,2,3',
            '--incast-total-mode', 'per-source',
            '--rail-phase', 'aligned',
        ] + extra
        for policy, extra in variants
    ]


def _joint(args: argparse.Namespace) -> list[list[str]]:
    variants = (
        ('off', []),
        ('active', []),
        ('incast', ['--incast-rail-overlap', '2.0']),
        ('incast', [
            '--incast-rail-overlap', '2.0',
            '--incast-weighted-quotas',
        ]),
    )
    commands = [
        _base(args, 'expert-rail-incast', policy) + [
            '--totals', args.joint_totals,
            '--rail-alphas', args.joint_alphas,
            '--expert-alphas', '0',
            '--source-alphas', '0',
            '--fan-ins', '3',
            '--incast-total-mode', 'per-source',
            '--rail-phase', 'aligned',
        ] + extra
        for policy, extra in variants
    ]
    if args.include_rotated:
        commands.extend(
            _base(args, 'expert-rail-incast', policy) + [
                '--totals', args.rotated_total,
                '--rail-alphas', args.rotated_alpha,
                '--expert-alphas', '0',
                '--source-alphas', '0',
                '--fan-ins', '3',
                '--incast-total-mode', 'per-source',
                '--rail-phase', 'rotated',
            ] + extra
            for policy, extra in variants
        )
    return commands


def _fast_compare(args: argparse.Namespace) -> list[list[str]]:
    """FAST-style global weighted permutations on the EchoP data plane."""
    return [
        _base(args, 'balanced-alltoall', 'active') + [
            '--totals', args.fast_total,
            '--rail-alphas', args.fast_rail_alphas,
            '--expert-alphas', '0',
            '--source-alphas', args.fast_source_alphas,
            '--fan-ins', '3',
            '--incast-total-mode', 'per-source',
            '--rail-phase', 'aligned',
            '--alltoall-flow-shape', 'directed-zipf',
            '--pairwise-peer-budget', '1',
            '--pairwise-planner', planner,
        ]
        for planner in ('cyclic-local', 'fast-global')
    ]


def _auto_calibration(args: argparse.Namespace) -> list[list[str]]:
    """Frozen Expert-to-Node-to-Rail calibration regimes."""
    variants = (
        ('off', []),
        ('active', []),
        ('incast', ['--incast-rail-overlap', '2.0']),
    )
    commands = []
    for policy, extra in variants:
        commands.append(
            _base(args, 'balanced-alltoall', policy) + [
                '--num-qps', str(args.auto_anchor_qps),
                '--totals', args.auto_anchor_total,
                '--rail-alphas', '0',
                '--expert-alphas', '0',
                '--source-alphas', '0',
                '--fan-ins', '3',
                '--incast-total-mode', 'per-source',
                '--rail-phase', 'aligned',
            ] + extra)
    for policy, extra in variants:
        commands.append(
            _base(args, 'incast', policy) + [
                '--totals', args.auto_node_total,
                '--rail-alphas', '0',
                '--expert-alphas', '0',
                '--source-alphas', args.auto_node_alphas,
                '--fan-ins', '3',
                '--incast-total-mode', 'per-source',
                '--rail-phase', 'aligned',
            ] + extra)
    for policy, extra in variants:
        commands.append(
            _base(args, 'balanced-alltoall', policy) + [
                '--totals', args.auto_rail_total,
                '--rail-alphas', args.auto_rail_alpha,
                '--expert-alphas', '0',
                '--source-alphas', '0',
                '--fan-ins', '3',
                '--incast-total-mode', 'per-source',
                '--rail-phase', 'rotated',
            ] + extra)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suite', choices=(
            'rail', 'incast', 'joint', 'four-node-core',
            'auto-calibration', 'fast-compare'),
        required=True)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--control-interval', type=int, default=32)
    parser.add_argument('--rail-totals', default='256,2048,8192,16384')
    parser.add_argument('--rail-alphas', default='0,0.75,1.5,2.25')
    parser.add_argument('--incast-totals', default='2048,8192,16384')
    parser.add_argument('--joint-totals', default='2048,8192,16384')
    parser.add_argument('--joint-alphas', default='0.75,1.5,2.25')
    parser.add_argument('--include-rotated', action='store_true')
    parser.add_argument('--rotated-total', default='8192')
    parser.add_argument('--rotated-alpha', default='1.5')
    parser.add_argument('--auto-anchor-total', default='16384')
    parser.add_argument('--auto-anchor-qps', type=int, default=9)
    parser.add_argument('--auto-node-total', default='8192')
    parser.add_argument('--auto-node-alphas', default='1,1.25,1.5')
    parser.add_argument('--auto-rail-total', default='16384')
    parser.add_argument('--auto-rail-alpha', default='2.25')
    parser.add_argument('--fast-total', default='65536')
    parser.add_argument(
        '--fast-source-alphas', default='0.5,0.75,1,1.25,1.5')
    parser.add_argument(
        '--fast-rail-alphas', default='1,1.25,1.75,2,2.25')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_processes <= 0 or args.iterations <= 0 or args.warmups < 0:
        parser.error('processes/iterations must be positive and warmups non-negative')

    if args.suite == 'rail':
        commands = _rail(args)
    elif args.suite == 'incast':
        commands = _incast(args)
    elif args.suite == 'joint':
        commands = _joint(args)
    elif args.suite == 'four-node-core':
        commands = _incast(args) + _joint(args)
    elif args.suite == 'fast-compare':
        commands = _fast_compare(args)
    else:
        commands = _auto_calibration(args)

    print('SUITE_CONFIG ' + json.dumps({
        'suite': args.suite,
        'num_commands': len(commands),
        'commands': commands,
    }), flush=True)
    if args.dry_run:
        return
    for index, command in enumerate(commands):
        print('SUITE_START ' + json.dumps({
            'index': index, 'command': command,
        }), flush=True)
        completed = subprocess.run(command, check=False)
        print('SUITE_END ' + json.dumps({
            'index': index, 'returncode': completed.returncode,
        }), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
