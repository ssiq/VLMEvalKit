#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import csv
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if 'LMUData' not in os.environ:
    lmu_data = Path(tempfile.gettempdir()) / 'LMUData'
    lmu_data.mkdir(parents=True, exist_ok=True)
    os.environ['LMUData'] = str(lmu_data)

import print_default_judge_models as default_judge_audit  # noqa: E402


DEFAULT_CSV = REPO_ROOT / 'default_judge_models.csv'
DEFAULT_COLUMN = 'codex_inferred_judge_model'
MODEL_RESULT_COLUMNS = {
    'default_judge_model',
    'run_judge_model',
    'class_default_judge',
    'evaluate_default_judge_model',
    'forced_override_judge_model',
    'runtime_probe_judge_model',
    'runtime_probe_status',
    'runtime_probe_error',
    'effective_default_judge_model',
    'effective_default_source',
    'judge_requirement',
    'judge_override_broken',
    'judge_key',
    'override_behavior',
    'notes',
    DEFAULT_COLUMN,
}
CODEX_OUTPUT_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'model': {'type': 'string'},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
        'source': {'type': 'string'},
    },
    'required': ['model', 'confidence', 'source'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Call codex exec for CSV rows that need review and write the '
            'inferred judge model to a new CSV column.'
        )
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV,
        help=(
            'CSV to read. Defaults to default_judge_models.csv. When --output '
            'is omitted, this file is updated in place and used for resume.'
        ),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'CSV to write. Defaults to overwriting --csv atomically. For '
            'resumable runs against changing input CSVs, keep using the same '
            '--output path; if it already exists, it is read as the resume source.'
        ),
    )
    parser.add_argument(
        '--column',
        default=DEFAULT_COLUMN,
        help=f'Column to write. Defaults to {DEFAULT_COLUMN}.',
    )
    parser.add_argument(
        '--data',
        nargs='+',
        default=None,
        help='Only process these dataset names.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only process the first N selected rows.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all matching rows instead of only rows that need review.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute rows even when the output column is already non-empty.',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Concurrent codex exec processes. Defaults to 4.',
    )
    parser.add_argument(
        '--codex-bin',
        default='codex',
        help='Codex CLI executable. Defaults to codex.',
    )
    parser.add_argument(
        '--codex-model',
        default=None,
        help='Optional model argument passed to codex exec -m.',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=900,
        help='Timeout per codex exec call in seconds. Defaults to 900.',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='Rewrite output CSV after every N completed rows. Defaults to 5.',
    )
    parser.add_argument(
        '--max-source-chars',
        type=int,
        default=120000,
        help='Maximum code-context characters per row. Defaults to 120000.',
    )
    parser.add_argument(
        '--keep-last-messages',
        type=Path,
        default=None,
        help='Directory to keep raw Codex last-message files for debugging.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print selected rows and the first prompt without calling Codex or writing CSV.',
    )
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv_atomic(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.',
        suffix='.tmp',
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def get_source_block(title, obj):
    if obj is None:
        return ''
    try:
        source_lines, start_line = inspect.getsourcelines(obj)
        source_file = inspect.getsourcefile(obj) or ''
    except (OSError, TypeError):
        return ''

    try:
        rel_path = Path(source_file).resolve().relative_to(REPO_ROOT)
    except (OSError, ValueError):
        rel_path = Path(source_file).name
    source = textwrap.dedent(''.join(source_lines)).rstrip()
    return (
        f'### {title}\n'
        f'File: {rel_path}:{start_line}\n'
        f'```python\n{source}\n```'
    )


def unwrap_static_function(dataset_cls, name):
    try:
        attr = inspect.getattr_static(dataset_cls, name)
    except AttributeError:
        return None
    if isinstance(attr, (classmethod, staticmethod)):
        return attr.__func__
    if inspect.isfunction(attr):
        return attr
    func = getattr(attr, '__func__', None)
    return func if inspect.isfunction(func) else None


def build_dataset_resolver():
    (
        get_judge_dataset_name,
        get_judge_kwargs,
        load_data_config,
        load,
        dataset_module,
        DATASET_CLASSES,
        DATASET_TYPE,
        SUPPORTED_DATASETS,
        supported_video_datasets,
    ) = default_judge_audit.import_run_interfaces()

    dataset_class_map = default_judge_audit.get_supported_dataset_class_map(DATASET_CLASSES)
    return {
        'get_judge_dataset_name': get_judge_dataset_name,
        'get_judge_kwargs': get_judge_kwargs,
        'dataset_module': dataset_module,
        'supported_video_datasets': supported_video_datasets,
        'dataset_class_map': dataset_class_map,
    }


def build_run_context(resolver):
    required_blocks = [
        ('run.get_judge_dataset_name', resolver['get_judge_dataset_name']),
        ('run.get_judge_kwargs', resolver['get_judge_kwargs']),
    ]
    blocks = []
    missing = []
    for title, obj in required_blocks:
        block = get_source_block(title, obj)
        if block:
            blocks.append(block)
        else:
            missing.append(title)
    if missing:
        raise RuntimeError(
            'Failed to inspect required run.py source blocks: '
            + ', '.join(missing)
        )
    return '\n\n'.join(blocks)


def resolve_dataset_class(row, resolver):
    dataset_name = row['dataset']
    dataset_cls, resolved_dataset = default_judge_audit.get_dataset_class_and_resolved_name(
        dataset_name,
        {},
        resolver['dataset_module'],
        resolver['supported_video_datasets'],
        resolver['dataset_class_map'],
    )
    if dataset_cls is None and row.get('resolved_dataset'):
        dataset_cls, resolved_dataset = default_judge_audit.get_dataset_class_and_resolved_name(
            row['resolved_dataset'],
            {},
            resolver['dataset_module'],
            resolver['supported_video_datasets'],
            resolver['dataset_class_map'],
        )
    return dataset_cls, resolved_dataset


def build_code_context(row, resolver, run_context):
    dataset_cls, resolved_dataset = resolve_dataset_class(row, resolver)
    blocks = [run_context]
    class_info = {
        'class': dataset_cls.__name__ if dataset_cls is not None else '',
        'module': dataset_cls.__module__ if dataset_cls is not None else '',
        'TYPE': getattr(dataset_cls, 'TYPE', '') if dataset_cls is not None else '',
        'MODALITY': getattr(dataset_cls, 'MODALITY', '') if dataset_cls is not None else '',
        'DEFAULT_JUDGE': getattr(dataset_cls, 'DEFAULT_JUDGE', None) if dataset_cls is not None else None,
        'resolved_dataset': resolved_dataset,
    }
    blocks.append('### Dataset class metadata\n```json\n' + json.dumps(
        class_info, ensure_ascii=False, indent=2, default=str
    ) + '\n```')

    if dataset_cls is None:
        return '\n\n'.join(blocks)

    for method_name in ('supported_datasets', '__init__', 'evaluate'):
        block = get_source_block(
            f'{dataset_cls.__name__}.{method_name}',
            unwrap_static_function(dataset_cls, method_name),
        )
        if block:
            blocks.append(block)

    evaluate_func = unwrap_static_function(dataset_cls, 'evaluate')
    if evaluate_func is not None:
        for method_name in default_judge_audit.delegated_evaluate_methods(
            evaluate_func, row.get('resolved_dataset') or resolved_dataset
        ):
            block = get_source_block(
                f'{dataset_cls.__name__}.{method_name}',
                unwrap_static_function(dataset_cls, method_name),
            )
            if block:
                blocks.append(block)

    return '\n\n'.join(blocks)


def truncate_context(context, max_chars):
    if len(context) <= max_chars:
        return context
    omitted = len(context) - max_chars
    return (
        context[:max_chars]
        + f'\n\n### Context truncated\nOmitted {omitted} trailing characters.'
    )


def current_audit_summary(row, output_column):
    excluded = {output_column}
    keys = [
        'dataset',
        'resolved_dataset',
        'dataset_type',
        'dataset_class',
        'source',
        'default_judge_model',
        'run_judge_model',
        'evaluate_default_judge_model',
        'forced_override_judge_model',
        'runtime_probe_judge_model',
        'runtime_probe_status',
        'effective_default_source',
        'judge_requirement',
        'judge_override_broken',
        'notes',
    ]
    return {key: row.get(key, '') for key in keys if key not in excluded and row.get(key, '')}


def build_prompt(row, resolver, run_context, max_source_chars, output_column):
    code_context = build_code_context(row, resolver, run_context)
    code_context = truncate_context(code_context, max_source_chars)
    summary = current_audit_summary(row, output_column)
    return f"""
You are reviewing one VLMEvalKit default judge audit row.

Goal:
- Infer the judge model used when entering from run.py with this dataset and no --judge.
- run.py calls get_judge_kwargs(..., args) with args.judge=None, then evaluate(eval_file, **judge_kwargs) runs.
- Return the model name(s) that evaluate will finally use for judging.
- If evaluate builds multiple judge models, return all model names joined with "|" in execution order.
- If exact matching is the judge behavior, return "exact_matching".
- If no judge is used when --judge is absent, return an empty string.

Current audit row:
```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

Relevant source code:
{code_context}

Return only JSON matching this schema:
{{"model": "<model or model1|model2 or empty string>", "confidence": "high|medium|low", "source": "<brief source-based reason>"}}
""".strip()


def parse_json_object(text):
    text = (text or '').strip()
    if not text:
        raise ValueError('empty response')
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def normalize_model_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        normalized = [normalize_model_value(item) for item in value]
        return '|'.join(item for item in normalized if item)
    value = str(value).strip()
    if value.lower() in {'none', 'null', 'no judge', 'no_judge', 'n/a', 'not used'}:
        return ''
    return value


def validate_codex_result(data):
    if not isinstance(data, dict):
        return '__ERROR__:InvalidJSON:codex response is not a JSON object'
    required = {'model', 'confidence', 'source'}
    missing = sorted(required - set(data))
    if missing:
        return '__ERROR__:MissingField:codex response missing ' + '|'.join(missing)
    if not isinstance(data['model'], str):
        return '__ERROR__:InvalidField:codex response model is not a string'
    if not isinstance(data['confidence'], str):
        return '__ERROR__:InvalidField:codex response confidence is not a string'
    if data['confidence'] not in {'high', 'medium', 'low'}:
        return '__ERROR__:InvalidField:codex response confidence is invalid'
    if not isinstance(data['source'], str):
        return '__ERROR__:InvalidField:codex response source is not a string'
    return ''


def close_process_transport(process):
    transport = getattr(process, '_transport', None)
    if transport is None:
        warnings.warn(
            'asyncio subprocess transport is unavailable; cleanup may be incomplete',
            RuntimeWarning,
            stacklevel=2,
        )
        return
    transport.close()


async def kill_process(process):
    if process is None or process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
        await asyncio.wait_for(process.wait(), timeout=5)


def row_needs_review(row):
    status = row.get('runtime_probe_status', '')
    notes = row.get('notes', '')
    if status.startswith('error:') or status in {'timeout', 'captured_without_model'}:
        return True
    if 'runtime_probe_differs_from_static=' in notes:
        return True
    if row.get('judge_requirement') in {'requires_judge', 'unknown', 'optional_judge'}:
        return True
    if row.get('judge_override_broken') == 'true':
        return True
    if row.get('forced_override_judge_model'):
        return True
    return False


def select_row_indices(rows, args):
    wanted = set(args.data or [])
    selected = []
    for idx, row in enumerate(rows):
        if wanted and row.get('dataset') not in wanted:
            continue
        if not args.all and not row_needs_review(row):
            continue
        if not args.force:
            current = row.get(args.column, '')
            if current and not current.startswith('__ERROR__:'):
                continue
        selected.append(idx)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def build_codex_command(args, schema_path, last_message_path):
    codex_bin = shutil.which(args.codex_bin) or args.codex_bin
    command = [
        codex_bin,
        'exec',
        '--ephemeral',
        '--sandbox',
        'read-only',
        '--output-schema',
        str(schema_path),
        '--output-last-message',
        str(last_message_path),
        '--color',
        'never',
        '-C',
        str(REPO_ROOT),
    ]
    if args.codex_model:
        command.extend(['-m', args.codex_model])
    command.append('-')
    return command


async def run_codex_one(args, schema_path, prompt, row_index, dataset_name):
    if args.keep_last_messages:
        args.keep_last_messages.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_name)
        last_message_path = args.keep_last_messages / f'{row_index:04d}_{safe_name}.json'
    else:
        fd, tmp_name = tempfile.mkstemp(prefix='codex-last-message-', suffix='.json')
        os.close(fd)
        last_message_path = Path(tmp_name)

    command = build_codex_command(args, schema_path, last_message_path)
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT),
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(prompt.encode()),
            timeout=args.timeout,
        )
        if process.returncode != 0:
            message = (stderr.decode(errors='replace') or stdout.decode(errors='replace'))
            message = message.replace('\n', ' ')[:240]
            return f'__ERROR__:CodexExit{process.returncode}:{message}'

        if last_message_path.exists():
            raw = last_message_path.read_text()
        else:
            raw = stdout.decode(errors='replace')
        data = parse_json_object(raw)
        validation_error = validate_codex_result(data)
        if validation_error:
            return validation_error
        return normalize_model_value(data.get('model'))
    except asyncio.TimeoutError:
        if process:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    return (
                        '__ERROR__:Timeout:codex exec exceeded '
                        f'{args.timeout} seconds and did not exit after SIGKILL'
                    )
        return f'__ERROR__:Timeout:codex exec exceeded {args.timeout} seconds'
    except asyncio.CancelledError:
        await kill_process(process)
        raise
    except Exception as err:
        await kill_process(process)
        message = str(err).replace('\n', ' ')[:240]
        return f'__ERROR__:{err.__class__.__name__}:{message}'
    finally:
        if process is not None:
            close_process_transport(process)
        if not args.keep_last_messages:
            try:
                last_message_path.unlink()
            except OSError:
                pass


async def run_inference(args, rows, fieldnames):
    resolver = build_dataset_resolver()
    run_context = build_run_context(resolver)
    indices = select_row_indices(rows, args)
    if not indices:
        print('No rows selected.', file=sys.stderr)
        return

    output_path = args.output or args.csv
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0
    started_at = time.time()

    with tempfile.TemporaryDirectory(prefix='codex-judge-schema-') as tmp_dir:
        schema_path = Path(tmp_dir) / 'schema.json'
        schema_path.write_text(json.dumps(CODEX_OUTPUT_SCHEMA, indent=2))

        async def worker(row_index):
            async with semaphore:
                row = rows[row_index]
                prompt = build_prompt(
                    row,
                    resolver,
                    run_context,
                    args.max_source_chars,
                    args.column,
                )
                model = await run_codex_one(
                    args,
                    schema_path,
                    prompt,
                    row_index,
                    row.get('dataset', f'row-{row_index}'),
                )
                return row_index, model

        tasks = [asyncio.create_task(worker(idx)) for idx in indices]
        for task in asyncio.as_completed(tasks):
            row_index, model = await task
            rows[row_index][args.column] = model
            completed += 1
            dataset = rows[row_index].get('dataset', f'row-{row_index}')
            elapsed = time.time() - started_at
            print(
                f'[{completed}/{len(indices)}] {dataset}: {model!r} elapsed={elapsed:.1f}s',
                file=sys.stderr,
            )
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                write_csv_atomic(output_path, rows, fieldnames)

    write_csv_atomic(output_path, rows, fieldnames)


def main():
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError('--concurrency must be positive')
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')

    input_path = args.csv
    if args.output is not None and args.output.exists():
        input_path = args.output
        print(f'Resuming from existing output CSV: {input_path}', file=sys.stderr)

    rows, fieldnames = read_csv(input_path)
    if args.column not in fieldnames:
        fieldnames.append(args.column)
        for row in rows:
            row.setdefault(args.column, '')
    elif args.output is None:
        completed = sum(
            1
            for row in rows
            if row.get(args.column, '') and not row.get(args.column, '').startswith('__ERROR__:')
        )
        if completed:
            print(
                f'Resuming in place from {args.csv}: {completed} rows already have '
                f'{args.column}. Use --force to recompute them, or use a stable '
                '--output path when the input CSV changes.',
                file=sys.stderr,
            )

    if args.dry_run:
        resolver = build_dataset_resolver()
        run_context = build_run_context(resolver)
        indices = select_row_indices(rows, args)
        print('selected_rows:', len(indices))
        print('selected_datasets:', [rows[idx].get('dataset') for idx in indices[:20]])
        if indices:
            prompt = build_prompt(
                rows[indices[0]],
                resolver,
                run_context,
                args.max_source_chars,
                args.column,
            )
            print('\n--- first prompt ---')
            print(prompt)
        return

    asyncio.run(run_inference(args, rows, fieldnames))


if __name__ == '__main__':
    main()
