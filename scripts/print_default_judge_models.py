#!/usr/bin/env python3
import argparse
import ast
import csv
import inspect
import io
import json
import logging
import os
import signal
import sys
import tempfile
import textwrap
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if 'LMUData' not in os.environ:
    lmu_data = Path(tempfile.gettempdir()) / 'LMUData'
    lmu_data.mkdir(parents=True, exist_ok=True)
    os.environ['LMUData'] = str(lmu_data)


FIELDNAMES = [
    'dataset',
    'resolved_dataset',
    'dataset_type',
    'dataset_class',
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
    'source',
    'notes',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Print the judge model finally selected by evaluate() when entering '
            'through run.py without --judge.'
        )
    )
    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        default=None,
        help='Dataset names to print. Defaults to all supported datasets.',
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default=None,
        help='Same JSON dict string accepted by run.py --data-config.',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Same config file accepted by run.py --config.',
    )
    parser.add_argument(
        '--judge-args',
        type=str,
        default=None,
        help='Same JSON dict string accepted by run.py --judge-args.',
    )
    parser.add_argument(
        '--probe-evaluate',
        action='store_true',
        help=(
            'Dynamically call evaluate() with mocked load/build_judge to capture '
            'the model finally passed to build_judge.'
        ),
    )
    parser.add_argument(
        '--probe-timeout',
        type=float,
        default=5.0,
        help='Seconds to allow each dynamic evaluate() probe. Defaults to 5.',
    )
    parser.add_argument(
        '--format',
        choices=['tsv', 'csv', 'json', 'table'],
        default='tsv',
        help='Output format. Defaults to tsv.',
    )
    video_shortcut_group = parser.add_mutually_exclusive_group()
    video_shortcut_group.add_argument(
        '--include-video-shortcuts',
        dest='include_video_shortcuts',
        action='store_true',
        default=True,
        help='Include video shortcut dataset names from video_dataset_config.py.',
    )
    video_shortcut_group.add_argument(
        '--no-video-shortcuts',
        dest='include_video_shortcuts',
        action='store_false',
        help='Only include names from SUPPORTED_DATASETS.',
    )
    parser.add_argument(
        '--sort',
        action='store_true',
        help='Sort rows by dataset name.',
    )
    return parser.parse_args()


def import_run_interfaces():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        from run import get_judge_dataset_name, get_judge_kwargs, load_data_config
        import vlmeval.dataset as dataset_module
        from vlmeval.dataset import DATASET_CLASSES, DATASET_TYPE, SUPPORTED_DATASETS
        from vlmeval.dataset.video_dataset_config import supported_video_datasets
        from vlmeval.smp import load
    finally:
        logging.disable(previous_disable_level)

    return (
        get_judge_dataset_name,
        get_judge_kwargs,
        load_data_config,
        load,
        dataset_module,
        DATASET_CLASSES,
        DATASET_TYPE,
        SUPPORTED_DATASETS,
        supported_video_datasets,
    )


def make_run_args(args):
    return SimpleNamespace(
        judge_api_nproc=None,
        api_nproc=32,
        judge_retry=None,
        retry=6,
        verbose=False,
        judge_timeout=600,
        judge_args=args.judge_args,
        judge_base_url=None,
        judge_key=None,
        judge=None,
        use_verifier=False,
        use_vllm=False,
    )


def unique_in_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def format_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return '|'.join(str(item) for item in value)
    return str(value)


def join_unique(items):
    return ';'.join(unique_in_order([item for item in items if item]))


def get_supported_dataset_class_map(data_set_classes):
    mapping = {}
    for dataset_cls in data_set_classes:
        try:
            supported = dataset_cls.supported_datasets()
        except Exception:
            supported = []
        for dataset_name in supported:
            mapping.setdefault(dataset_name, dataset_cls)
    return mapping


def get_factory_class_and_dataset(factory, dataset_name):
    dataset_cls = getattr(factory, 'func', None)
    resolved_dataset = getattr(factory, 'keywords', {}).get('dataset', dataset_name)
    return dataset_cls, resolved_dataset


def get_video_shortcut_type(dataset_name, supported_video_datasets, DATASET_TYPE):
    factory = supported_video_datasets[dataset_name]
    dataset_cls = getattr(factory, 'func', None)
    if dataset_cls is not None and hasattr(dataset_cls, 'TYPE'):
        return dataset_cls.TYPE

    base_dataset = getattr(factory, 'keywords', {}).get('dataset', dataset_name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DATASET_TYPE(base_dataset)


def get_config_dataset_class(
    dataset_name, config_data, dataset_module, supported_video_datasets, dataset_class_map
):
    config = config_data.get(dataset_name, {})
    cls_name = config.get('class')
    resolved_dataset = config.get('dataset', dataset_name)
    if cls_name and hasattr(dataset_module, cls_name):
        return getattr(dataset_module, cls_name), resolved_dataset

    if resolved_dataset in supported_video_datasets:
        return get_factory_class_and_dataset(
            supported_video_datasets[resolved_dataset], resolved_dataset
        )

    return dataset_class_map.get(resolved_dataset), resolved_dataset


def get_dataset_class_and_resolved_name(
    dataset_name, config_data, dataset_module, supported_video_datasets, dataset_class_map
):
    if dataset_name in config_data:
        return get_config_dataset_class(
            dataset_name, config_data, dataset_module, supported_video_datasets, dataset_class_map
        )

    if dataset_name in supported_video_datasets:
        return get_factory_class_and_dataset(supported_video_datasets[dataset_name], dataset_name)

    return dataset_class_map.get(dataset_name), dataset_name


def get_config_dataset_type(
    dataset_name, config_data, dataset_module, supported_video_datasets, DATASET_TYPE
):
    config = config_data.get(dataset_name, {})
    cls_name = config.get('class')
    if cls_name and hasattr(dataset_module, cls_name):
        dataset_cls = getattr(dataset_module, cls_name)
        if hasattr(dataset_cls, 'TYPE'):
            return dataset_cls.TYPE

    base_dataset = config.get('dataset', dataset_name)
    if base_dataset in supported_video_datasets:
        return get_video_shortcut_type(base_dataset, supported_video_datasets, DATASET_TYPE)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DATASET_TYPE(base_dataset)


def get_dataset_type(
    dataset_name, config_data, dataset_module, supported_video_datasets, DATASET_TYPE
):
    if dataset_name in config_data:
        return get_config_dataset_type(
            dataset_name, config_data, dataset_module, supported_video_datasets, DATASET_TYPE
        )
    if dataset_name in supported_video_datasets:
        return get_video_shortcut_type(dataset_name, supported_video_datasets, DATASET_TYPE)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DATASET_TYPE(dataset_name)


def get_dataset_source(
    dataset_name, data_config, config_data, supported_datasets, supported_video_datasets
):
    if dataset_name in config_data:
        return 'config_file'
    if dataset_name in data_config:
        return 'data_config'
    if dataset_name in supported_video_datasets:
        return 'video_shortcut'
    if dataset_name in supported_datasets:
        return 'supported_dataset'
    return 'unknown'


def get_dataset_names(args, data_config, config_data, supported_datasets, supported_video_datasets):
    if args.data:
        return unique_in_order(args.data)

    if args.config:
        dataset_names = list(config_data)
        if args.sort:
            dataset_names.sort()
        return dataset_names

    dataset_names = list(supported_datasets)
    if args.include_video_shortcuts:
        dataset_names.extend(supported_video_datasets)
    dataset_names.extend(data_config)
    dataset_names = unique_in_order(dataset_names)
    if args.sort:
        dataset_names.sort()
    return dataset_names


def load_configs(args, load_data_config, load):
    if args.config and args.data_config:
        raise ValueError('--config and --data-config cannot be used together.')

    data_config = load_data_config(args.data_config)
    config_data = {}
    if args.config:
        config = load(args.config)
        config_data = config.get('data', {})
        if not isinstance(config_data, dict):
            raise ValueError('--config must contain a dict under the `data` key.')
        data_config = {}
    return data_config, config_data


def get_static_evaluate_func(dataset_cls):
    if dataset_cls is None:
        return None
    try:
        attr = inspect.getattr_static(dataset_cls, 'evaluate')
    except AttributeError:
        return None
    if isinstance(attr, (classmethod, staticmethod)):
        return attr.__func__
    if inspect.isfunction(attr):
        return attr
    func = getattr(attr, '__func__', None)
    return func if inspect.isfunction(func) else None


def ast_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def is_judge_kwargs(node):
    return isinstance(node, ast.Name) and node.id == 'judge_kwargs'


def subscript_key(node):
    if not isinstance(node, ast.Subscript) or not is_judge_kwargs(node.value):
        return None
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def judge_kwargs_call(node):
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or not is_judge_kwargs(func.value):
        return None
    if func.attr not in {'get', 'pop', 'setdefault'}:
        return None
    if not node.args:
        return None
    key = ast_string(node.args[0])
    if key not in {'model', 'judge_model'}:
        return None
    default = node.args[1] if len(node.args) > 1 else None
    return func.attr, key, default


def is_none_node(node):
    return node is None or (isinstance(node, ast.Constant) and node.value is None)


def resolve_model_node(node, dataset_cls):
    if node is None:
        return None, ''
    if isinstance(node, ast.Constant):
        return node.value, 'literal'
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for item in node.elts:
            value, _ = resolve_model_node(item, dataset_cls)
            values.append(value)
        return values, 'literal'
    if isinstance(node, ast.Attribute) and node.attr == 'DEFAULT_JUDGE':
        return getattr(dataset_cls, 'DEFAULT_JUDGE', None), 'DEFAULT_JUDGE'
    return None, ''


def has_model_missing_check(node):
    if isinstance(node, ast.Compare):
        left_call = judge_kwargs_call(node.left)
        if left_call and left_call[1] == 'model':
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.Is) and isinstance(comparator, ast.Constant):
                    if comparator.value is None:
                        return True
    for child in ast.iter_child_nodes(node):
        if has_model_missing_check(child):
            return True
    return False


def is_inside_missing_model_guard(node):
    parent = getattr(node, 'parent', None)
    while parent is not None:
        if isinstance(parent, ast.If) and has_model_missing_check(parent.test):
            return True
        parent = getattr(parent, 'parent', None)
    return False


class EvaluateJudgeAnalyzer(ast.NodeVisitor):

    def __init__(self, dataset_cls, start_line):
        self.dataset_cls = dataset_cls
        self.start_line = start_line
        self.fallbacks = []
        self.assignments = []
        self.required_model_lines = []
        self.optional_model_lines = []
        self.model_key_lines = []
        self.build_judge_kwargs_lines = []
        self.explicit_model_build_judge_kwargs_lines = []
        self.unresolved_model_default_lines = []
        self.nonstandard_keys = set()

    def abs_line(self, node):
        return self.start_line + getattr(node, 'lineno', 1) - 1

    def add_fallback(self, key, method, model, model_source, node):
        if model is None:
            return
        self.fallbacks.append({
            'key': key,
            'method': method,
            'model': format_value(model),
            'model_source': model_source,
            'line': self.abs_line(node),
        })
        if key != 'model':
            self.nonstandard_keys.add(key)

    def visit_Call(self, node):
        call = judge_kwargs_call(node)
        if call is not None:
            method, key, default = call
            self.model_key_lines.append(self.abs_line(node))
            if key != 'model':
                self.nonstandard_keys.add(key)
            if key == 'model' and method == 'get' and is_none_node(default):
                self.optional_model_lines.append(self.abs_line(node))
            if default is not None:
                model, model_source = resolve_model_node(default, self.dataset_cls)
                self.add_fallback(key, method, model, model_source, node)
                if model is None and not is_none_node(default):
                    self.unresolved_model_default_lines.append(self.abs_line(node))

        if self.is_build_judge_call(node):
            has_judge_kwargs = any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == 'judge_kwargs'
                for keyword in node.keywords
            )
            has_explicit_model = any(keyword.arg == 'model' for keyword in node.keywords)
            if has_judge_kwargs and has_explicit_model:
                self.explicit_model_build_judge_kwargs_lines.append(self.abs_line(node))
            elif has_judge_kwargs:
                self.build_judge_kwargs_lines.append(self.abs_line(node))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        key = subscript_key(node)
        if key == 'model' and isinstance(node.ctx, ast.Load):
            self.model_key_lines.append(self.abs_line(node))
            self.required_model_lines.append(self.abs_line(node))
        self.generic_visit(node)

    @staticmethod
    def is_build_judge_call(node):
        func = node.func
        return isinstance(func, ast.Name) and func.id == 'build_judge'

    def visit_Assign(self, node):
        for target in node.targets:
            key = subscript_key(target)
            if key != 'model':
                continue

            rhs_call = judge_kwargs_call(node.value)
            if rhs_call is not None and rhs_call[1] == 'model' and rhs_call[2] is not None:
                model, model_source = resolve_model_node(rhs_call[2], self.dataset_cls)
                self.add_fallback('model', 'assign_get', model, model_source, node)
                continue

            model, model_source = resolve_model_node(node.value, self.dataset_cls)
            if model is None:
                continue
            behavior = 'if_missing' if is_inside_missing_model_guard(node) else 'forced_override'
            self.assignments.append({
                'model': format_value(model),
                'model_source': model_source,
                'behavior': behavior,
                'line': self.abs_line(node),
            })
        self.generic_visit(node)


def empty_analysis():
    return {
        'fallbacks': [],
        'assignments': [],
        'required_model_lines': [],
        'optional_model_lines': [],
        'model_key_lines': [],
        'build_judge_kwargs_lines': [],
        'explicit_model_build_judge_kwargs_lines': [],
        'unresolved_model_default_lines': [],
        'nonstandard_keys': set(),
        'evaluate_owner': '',
    }


def merge_analysis(base, extra):
    base['fallbacks'].extend(extra['fallbacks'])
    base['assignments'].extend(extra['assignments'])
    base['required_model_lines'].extend(extra['required_model_lines'])
    base['optional_model_lines'].extend(extra['optional_model_lines'])
    base['model_key_lines'].extend(extra['model_key_lines'])
    base['build_judge_kwargs_lines'].extend(extra['build_judge_kwargs_lines'])
    base['explicit_model_build_judge_kwargs_lines'].extend(
        extra['explicit_model_build_judge_kwargs_lines']
    )
    base['unresolved_model_default_lines'].extend(extra['unresolved_model_default_lines'])
    base['nonstandard_keys'].update(extra['nonstandard_keys'])
    owners = [base.get('evaluate_owner', ''), extra.get('evaluate_owner', '')]
    base['evaluate_owner'] = ';'.join(unique_in_order([owner for owner in owners if owner]))
    return base


def is_self_dataset_name(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'dataset_name'
        and isinstance(node.value, ast.Name)
        and node.value.id in {'self', 'cls'}
    )


def dataset_names_in_test(node):
    names = set()
    if isinstance(node, ast.Compare):
        left_is_dataset = is_self_dataset_name(node.left)
        for comparator in node.comparators:
            right_name = ast_string(comparator)
            if left_is_dataset and right_name:
                names.add(right_name)
            if is_self_dataset_name(comparator):
                left_name = ast_string(node.left)
                if left_name:
                    names.add(left_name)
    for child in ast.iter_child_nodes(node):
        names.update(dataset_names_in_test(child))
    return names


def delegated_methods_in_nodes(nodes):
    methods = set()
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name) or func.value.id not in {'self', 'cls'}:
                continue
            if not func.attr.startswith('evaluate_'):
                continue
            if any(keyword.arg is None and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == 'judge_kwargs' for keyword in child.keywords):
                methods.add(func.attr)
    return methods


def delegated_evaluate_methods(func, dataset_name):
    try:
        source_lines, _ = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(textwrap.dedent(''.join(source_lines)).lstrip())
    except SyntaxError:
        return []
    matching_methods = set()
    dataset_condition_seen = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = dataset_names_in_test(node.test)
        if names:
            dataset_condition_seen = True
            if dataset_name in names:
                matching_methods.update(delegated_methods_in_nodes(node.body))

    if matching_methods:
        return sorted(matching_methods)
    if dataset_condition_seen:
        return []
    return sorted(delegated_methods_in_nodes([tree]))


def analyze_func_defaults(func, dataset_cls):
    result = empty_analysis()
    if func is None:
        return result

    try:
        source_lines, start_line = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return result

    source = textwrap.dedent(''.join(source_lines)).lstrip()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

    analyzer = EvaluateJudgeAnalyzer(dataset_cls, start_line)
    analyzer.visit(tree)
    result.update({
        'fallbacks': analyzer.fallbacks,
        'assignments': analyzer.assignments,
        'required_model_lines': analyzer.required_model_lines,
        'optional_model_lines': analyzer.optional_model_lines,
        'model_key_lines': analyzer.model_key_lines,
        'build_judge_kwargs_lines': analyzer.build_judge_kwargs_lines,
        'explicit_model_build_judge_kwargs_lines': analyzer.explicit_model_build_judge_kwargs_lines,
        'unresolved_model_default_lines': analyzer.unresolved_model_default_lines,
        'nonstandard_keys': analyzer.nonstandard_keys,
        'evaluate_owner': f'{func.__module__}.{func.__qualname__}',
    })
    return result


def analyze_evaluate_defaults(dataset_cls, dataset_name):
    func = get_static_evaluate_func(dataset_cls)
    if func is None:
        return empty_analysis()

    result = analyze_func_defaults(func, dataset_cls)
    has_judge_info = (
        result['fallbacks'] or result['assignments'] or result['required_model_lines']
        or result['nonstandard_keys']
    )
    if has_judge_info:
        return result

    for method_name in delegated_evaluate_methods(func, dataset_name):
        try:
            attr = inspect.getattr_static(dataset_cls, method_name)
        except AttributeError:
            continue
        if isinstance(attr, (classmethod, staticmethod)):
            method_func = attr.__func__
        else:
            method_func = attr if inspect.isfunction(attr) else getattr(attr, '__func__', None)
        merge_analysis(result, analyze_func_defaults(method_func, dataset_cls))
    return result


def resolve_effective_default(run_model, analysis):
    forced_models = [
        item['model'] for item in analysis['assignments']
        if item['behavior'] == 'forced_override'
    ]
    if forced_models:
        return join_unique(forced_models), 'evaluate_forced_override'
    if run_model:
        return run_model, 'run.py'

    fallback_models = [item['model'] for item in analysis['fallbacks']]
    if fallback_models:
        return join_unique(fallback_models), 'evaluate_fallback'

    if_missing_models = [
        item['model'] for item in analysis['assignments']
        if item['behavior'] == 'if_missing'
    ]
    if if_missing_models:
        return join_unique(if_missing_models), 'evaluate_if_missing'

    if analysis['optional_model_lines']:
        return '', 'optional_judge'
    if analysis['required_model_lines'] or analysis['build_judge_kwargs_lines']:
        return '', 'requires_run_model'
    return '', ''


def get_evaluate_default_models(analysis):
    fallback_models = [item['model'] for item in analysis['fallbacks']]
    if_missing_models = [
        item['model'] for item in analysis['assignments']
        if item['behavior'] == 'if_missing'
    ]
    return join_unique(fallback_models + if_missing_models)


class JudgeProbeCaptured(Exception):

    def __init__(self, args, kwargs):
        super().__init__('judge probe captured build_judge call')
        self.args_tuple = args
        self.kwargs = kwargs


class JudgeProbeTimeout(Exception):
    pass


def make_probe_dataframe(dataset_name):
    import pandas as pd

    row = {
        'index': 0,
        'qid': '0',
        'question': 'What is the answer?',
        'answer': 'A',
        'gt': 'A',
        'ground_truth': 'A',
        'prediction': 'A',
        'category': 'general',
        'sub_category': 'general',
        'task': 'documentQA',
        'image': '',
        'image_path': '/tmp/default_judge_probe.jpg',
        'A': 'A',
        'B': 'B',
        'C': 'C',
        'D': 'D',
        'option_a': 'A',
        'option_b': 'B',
        'option_c': 'C',
        'option_d': 'D',
        'choices': 'A\nB\nC\nD',
        'answer_type': 'text',
        'marking': 'A',
        'points': 1,
        'score': 1,
        'model_result': -1,
        'dataset': dataset_name,
    }
    return pd.DataFrame([row])


def fake_intermediate_file_path(eval_file, suffix, ext=None):
    stem = str(eval_file).rsplit('.', 1)[0]
    if ext:
        return f'{stem}{suffix}.{ext}'
    return f'{stem}{suffix}.xlsx'


def fake_track_progress(*args, **kwargs):
    tasks = kwargs.get('tasks')
    if tasks is None and args:
        tasks = args[0]
    try:
        return [None for _ in tasks]
    except TypeError:
        return []


def fake_file_size(*args, **kwargs):
    return 0


def fake_read_ok(*args, **kwargs):
    return True


def fake_decode_base64_to_image_file(*args, **kwargs):
    return None


def fake_download_file(*args, **kwargs):
    return None


@contextmanager
def disabled_logging():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


@contextmanager
def patched_probe_environment(fake_df):
    patches = []

    def add_patch(obj, name, value):
        if obj is None or not hasattr(obj, name):
            return
        patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def fake_build_judge(*args, **kwargs):
        raise JudgeProbeCaptured(args, kwargs)

    def fake_load(*args, **kwargs):
        return fake_df.copy()

    def fake_dump(*args, **kwargs):
        return None

    def fake_exists(path):
        return False

    def fake_path_exists(self):
        return False

    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (
            module_name == 'vlmeval.smp'
            or module_name.startswith('vlmeval.smp')
            or module_name.startswith('vlmeval.dataset')
        ):
            continue
        add_patch(module, 'build_judge', fake_build_judge)
        add_patch(module, 'load', fake_load)
        add_patch(module, 'dump', fake_dump)
        add_patch(module, 'get_intermediate_file_path', fake_intermediate_file_path)
        add_patch(module, 'track_progress_rich', fake_track_progress)
        add_patch(module, 'track_progress', fake_track_progress)
        add_patch(module, 'file_size', fake_file_size)
        add_patch(module, 'read_ok', fake_read_ok)
        add_patch(module, 'decode_base64_to_image_file', fake_decode_base64_to_image_file)
        add_patch(module, 'download_file', fake_download_file)

    add_patch(os.path, 'exists', fake_exists)
    add_patch(os, 'makedirs', lambda *args, **kwargs: None)
    add_patch(Path, 'exists', fake_path_exists)

    try:
        yield
    finally:
        for obj, name, old_value in reversed(patches):
            setattr(obj, name, old_value)


@contextmanager
def probe_timeout(seconds):
    if seconds is None or seconds <= 0:
        yield
        return

    def handle_timeout(signum, frame):
        raise JudgeProbeTimeout(f'probe timed out after {seconds} seconds')

    old_handler = signal.signal(signal.SIGALRM, handle_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def make_probe_instance(dataset_cls, dataset_name, fake_df):
    inst = object.__new__(dataset_cls)
    inst.dataset_name = dataset_name
    inst.data = fake_df.copy()
    inst.TYPE = getattr(dataset_cls, 'TYPE', '')
    inst.MODALITY = getattr(dataset_cls, 'MODALITY', '')
    inst.meta_only = True
    inst.skip_noimg = False
    inst.img_root = tempfile.gettempdir()
    return inst


def call_evaluate_for_probe(dataset_cls, dataset_name, fake_eval_file, judge_kwargs, fake_df):
    attr = inspect.getattr_static(dataset_cls, 'evaluate')
    if isinstance(attr, classmethod):
        return attr.__func__(dataset_cls, fake_eval_file, **judge_kwargs)
    if isinstance(attr, staticmethod):
        return attr.__func__(fake_eval_file, **judge_kwargs)
    if inspect.isfunction(attr):
        inst = make_probe_instance(dataset_cls, dataset_name, fake_df)
        return attr(inst, fake_eval_file, **judge_kwargs)
    func = getattr(attr, '__func__', None)
    if inspect.isfunction(func):
        inst = make_probe_instance(dataset_cls, dataset_name, fake_df)
        return func(inst, fake_eval_file, **judge_kwargs)
    raise TypeError('evaluate is not a callable function')


def normalize_probe_model(captured):
    model = captured.kwargs.get('model')
    if model is None and captured.args_tuple:
        model = captured.args_tuple[0]
    return format_value(model)


def probe_evaluate_judge_model(dataset_cls, dataset_name, judge_kwargs, timeout):
    if dataset_cls is None:
        return {
            'model': '',
            'status': 'skipped_no_dataset_class',
            'error': '',
        }

    try:
        inspect.getattr_static(dataset_cls, 'evaluate')
    except AttributeError:
        return {
            'model': '',
            'status': 'skipped_no_evaluate',
            'error': '',
        }

    fake_df = make_probe_dataframe(dataset_name)
    fake_eval_file = str(Path(tempfile.gettempdir()) / 'default_judge_probe.xlsx')
    try:
        with (
            patched_probe_environment(fake_df),
            probe_timeout(timeout),
            disabled_logging(),
            warnings.catch_warnings(),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter('ignore')
            call_evaluate_for_probe(
                dataset_cls,
                dataset_name,
                fake_eval_file,
                dict(judge_kwargs),
                fake_df,
            )
    except JudgeProbeCaptured as err:
        model = normalize_probe_model(err)
        return {
            'model': model,
            'status': 'captured' if model else 'captured_without_model',
            'error': '',
        }
    except JudgeProbeTimeout as err:
        return {
            'model': '',
            'status': 'timeout',
            'error': str(err),
        }
    except Exception as err:
        message = str(err).replace('\n', ' ')[:240]
        return {
            'model': '',
            'status': f'error:{err.__class__.__name__}',
            'error': message,
        }

    return {
        'model': '',
        'status': 'completed_without_build_judge',
        'error': '',
    }


def get_judge_requirement(effective_model, effective_source, analysis):
    if effective_model:
        return 'has_default'
    if effective_source == 'requires_run_model':
        return 'requires_judge'
    if effective_source == 'optional_judge':
        return 'optional_judge'
    if analysis['model_key_lines'] or analysis['build_judge_kwargs_lines']:
        return 'unknown'
    return 'no_judge_needed'


def judge_override_broken(analysis):
    return bool(analysis['nonstandard_keys'])


def build_notes(
    run_model,
    class_default,
    analysis,
    effective_model,
    effective_source,
    judge_requirement,
    override_broken,
    extra_notes=None,
):
    notes = []
    if extra_notes:
        notes.extend(extra_notes)
    fallback_models = get_evaluate_default_models(analysis)
    if run_model and fallback_models and run_model != fallback_models:
        notes.append(f'direct_evaluate_default={fallback_models}')
    if analysis['nonstandard_keys']:
        notes.append('uses_nonstandard_judge_key=' + '|'.join(sorted(analysis['nonstandard_keys'])))
    if override_broken:
        notes.append('judge_override_broken')
    forced_models = join_unique(
        item['model'] for item in analysis['assignments']
        if item['behavior'] == 'forced_override'
    )
    if forced_models:
        notes.append('evaluate_overrides_model')
    if class_default and class_default not in {effective_model, fallback_models}:
        notes.append('class_DEFAULT_JUDGE_not_effective_default')
    if effective_source == 'requires_run_model':
        lines = ','.join(str(line) for line in analysis['required_model_lines'])
        if lines:
            notes.append(f'evaluate_requires_model_at_lines={lines}')
        build_lines = ','.join(str(line) for line in analysis['build_judge_kwargs_lines'])
        if build_lines:
            notes.append(f'build_judge_requires_model_at_lines={build_lines}')
    if effective_source == 'optional_judge':
        lines = ','.join(str(line) for line in analysis['optional_model_lines'])
        notes.append(f'optional_judge_model_at_lines={lines}')
    if judge_requirement == 'unknown' and analysis['unresolved_model_default_lines']:
        lines = ','.join(str(line) for line in analysis['unresolved_model_default_lines'])
        notes.append(f'unresolved_model_default_at_lines={lines}')
    return ';'.join(notes)


def build_rows(args):
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
    ) = import_run_interfaces()

    data_config, config_data = load_configs(args, load_data_config, load)
    run_args = make_run_args(args)
    rows = []

    dataset_names = get_dataset_names(
        args, data_config, config_data, SUPPORTED_DATASETS, supported_video_datasets
    )
    supported_dataset_set = set(SUPPORTED_DATASETS)
    dataset_class_map = get_supported_dataset_class_map(DATASET_CLASSES)

    for dataset_name in dataset_names:
        class_config_data = config_data if args.config else data_config
        dataset_cls, resolved_dataset = get_dataset_class_and_resolved_name(
            dataset_name,
            class_config_data,
            dataset_module,
            supported_video_datasets,
            dataset_class_map,
        )
        dataset_type = get_dataset_type(
            dataset_name, class_config_data, dataset_module, supported_video_datasets, DATASET_TYPE
        )
        judge_dataset_name = get_judge_dataset_name(dataset_name, data_config)
        judge_kwargs = get_judge_kwargs(judge_dataset_name, dataset_type, run_args)
        run_model = judge_kwargs.get('model') or ''
        analysis = analyze_evaluate_defaults(dataset_cls, resolved_dataset)
        class_default = format_value(getattr(dataset_cls, 'DEFAULT_JUDGE', None))
        evaluate_default = get_evaluate_default_models(analysis)
        forced_override = join_unique(
            item['model'] for item in analysis['assignments']
            if item['behavior'] == 'forced_override'
        )
        effective_model, effective_source = resolve_effective_default(run_model, analysis)
        override_broken = judge_override_broken(analysis)
        probe_result = {
            'model': '',
            'status': 'disabled',
            'error': '',
        }
        extra_notes = []
        if args.probe_evaluate:
            probe_result = probe_evaluate_judge_model(
                dataset_cls,
                resolved_dataset,
                judge_kwargs,
                args.probe_timeout,
            )
            if probe_result['model'] and probe_result['model'] != effective_model:
                extra_notes.append(
                    'runtime_probe_differs_from_static='
                    f'{probe_result["model"]}!={effective_model or "empty"}'
                )
        default_model = probe_result['model'] or effective_model
        default_source = effective_source
        if probe_result['model'] and probe_result['model'] != effective_model:
            default_source = 'runtime_probe'
        judge_requirement = get_judge_requirement(default_model, default_source, analysis)
        if args.config and dataset_name != resolved_dataset:
            canonical_kwargs = get_judge_kwargs(resolved_dataset, dataset_type, run_args)
            canonical_run_model = canonical_kwargs.get('model') or ''
            if canonical_run_model != run_model:
                extra_notes.append(
                    'config_alias_changes_run_dispatch='
                    f'{canonical_run_model or "empty"}'
                )
        judge_keys = {'model'}
        judge_keys.update(analysis['nonstandard_keys'])
        override_behavior = join_unique(item['behavior'] for item in analysis['assignments'])
        rows.append({
            'dataset': dataset_name,
            'resolved_dataset': resolved_dataset,
            'dataset_type': dataset_type,
            'dataset_class': dataset_cls.__name__ if dataset_cls is not None else '',
            'default_judge_model': default_model,
            'run_judge_model': run_model,
            'class_default_judge': class_default,
            'evaluate_default_judge_model': evaluate_default,
            'forced_override_judge_model': forced_override,
            'runtime_probe_judge_model': probe_result['model'],
            'runtime_probe_status': probe_result['status'],
            'runtime_probe_error': probe_result['error'],
            'effective_default_judge_model': default_model,
            'effective_default_source': default_source,
            'judge_requirement': judge_requirement,
            'judge_override_broken': str(override_broken).lower(),
            'judge_key': '|'.join(sorted(judge_keys)),
            'override_behavior': override_behavior,
            'source': get_dataset_source(
                dataset_name, data_config, config_data, supported_dataset_set, supported_video_datasets
            ),
            'notes': build_notes(
                run_model,
                class_default,
                analysis,
                default_model,
                default_source,
                judge_requirement,
                override_broken,
                extra_notes,
            ),
        })

    return rows


def print_rows(rows, output_format):
    if output_format == 'json':
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    if output_format == 'table':
        from tabulate import tabulate
        print(tabulate(rows, headers='keys'))
        return

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=FIELDNAMES,
        dialect='excel-tab' if output_format == 'tsv' else 'excel',
    )
    writer.writeheader()
    writer.writerows(rows)


def main():
    args = parse_args()
    rows = build_rows(args)
    print_rows(rows, args.format)


if __name__ == '__main__':
    main()
