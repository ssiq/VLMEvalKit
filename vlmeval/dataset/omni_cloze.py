# flake8: noqa
import ast
import copy
import json
import os
import os.path as osp
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import portalocker

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .utils import build_judge
from .video_base import VideoBaseDataset


def _omni_cloze_prediction_text(prediction):
    if prediction is None:
        return ''
    try:
        if pd.isna(prediction):
            return ''
    except (TypeError, ValueError):
        pass
    return str(prediction)


def OmniCloze_auxeval(judge, line, modality='audio-visual', number=None, dataset_name='Omni-Cloze'):
    item = line.to_dict() if hasattr(line, 'to_dict') else dict(line)
    cloze = OmniCloze._parse_cloze(item['cloze'])
    prediction = _omni_cloze_prediction_text(item.get('prediction'))
    prompt = OmniCloze.build_judge_prompt(cloze, prediction, modality=modality, number=number)
    response = judge.generate([dict(type='text', value=prompt)], dataset=dataset_name)
    raw_response = response.get('prediction') if isinstance(response, dict) else response
    parsed = OmniCloze._parse_prediction(raw_response)
    if not OmniCloze._is_valid_prediction(parsed, number or OmniCloze.DEFAULT_BLANKS):
        parsed = OmniCloze._default_cloze_prediction(cloze)
    return {
        'cloze': cloze,
        'metadata': OmniCloze._parse_metadata(item.get('metadata')),
        'predicted_caption': prediction,
        'judge_raw_response': raw_response,
        'cloze_prediction': parsed,
    }


class OmniCloze(VideoBaseDataset):
    REPO_ID = 'BoJack/Omni-Cloze'
    DATA_FILE = 'omni_cloze.jsonl'
    TABLE_FILE = 'Omni-Cloze.tsv'
    VIDEO_ARCHIVE_GLOB = 'videos.part*.tar'
    TYPE = 'Video-VQA'
    MODALITY = 'VIDEO'
    DEFAULT_JUDGE_MODEL = 'gpt-4o-1120'
    DEFAULT_BLANKS = 30
    FAIL_MSG = 'Failed to obtain answer via API.'

    CAPTION_PROMPT = """
Provide a detailed description of the video.

It should explicitly include three sections:

1. A structured chronological storyline of **every noticeable audio and visual details**
2. A structured list of all visible text. For each text element, include start timestamp, end timestamp, the exact text content, the appearance characteristics. If no text appears, explicitly state so.
3. A structured speech-to-text transcription, include speaker（Corresponding to the character or voice‑over in Section 1, including their accent and tone）, exact spoken content, start timestamp, end timestamp, and speaking state (prosody, emotion, and style). If no speech appears, explicitly state so.

Aside from these three required sections, you are free to organize any additional content in any way you find helpful. This additional content can include global information about the entire video or localized information about specific moments. You may choose the topic of this extra content freely.

Output Format:

## Storyline

<xx:xx.xxx> - <xx:xx.xxx>
<an unstructured long paragraph in natural language describing what happened during this period, blending both audio and video details.>

<xx:xx.xxx> - <xx:xx.xxx>
<an unstructured long paragraph in natural language describing what happened during this period, blending both audio and video details.>

<xx:xx.xxx> - <xx:xx.xxx>
<an unstructured long paragraph in natural language describing what happened during this period, blending both audio and video details.>

...

## Visible Text

<xx:xx.xxx> - <xx:xx.xxx>
“<element>”: <appearance>
“<element>”: <appearance>

<xx:xx.xxx> - <xx:xx.xxx>
“<element>”: <appearance>
“<element>”: <appearance>
“<element>”: <appearance>

<xx:xx.xxx> - <xx:xx.xxx>
“<element>”: <appearance>

...

## Speakers and Transcript

Speaker profiles:
<speaker> - <profile>
<speaker> - <profile>
<speaker> - <profile>
...

<xx:xx.xxx> - <xx:xx.xxx>
Speaker: <speaker>
State: <description>
Content: “<content>”

<xx:xx.xxx> - <xx:xx.xxx>
Speaker: <speaker>
State: <description>
Content: “<content>”

<xx:xx.xxx> - <xx:xx.xxx>
Speaker: <speaker>
State: <description>
Content: “<content>”

...

## <another section>

<paragraphs>

## <another section>

<paragraphs>

...
""".strip()

    JUDGE_SYSTEM_PROMPT = "You are a helpful assistant."
    JUDGE_PROMPT_TEMPLATE = """
You will be given two inputs:
1. A cloze (fill-in-the-blank) task, consisting of a passage describing a scene, with {number} blanks. Each blank has 5 possible choices (A-D are specific contents, E means "not given").
2. A caption ( {modality} description of the scene).

Your task:
- Carefully read the caption and use it to determine the correct answer for each blank.
- If the caption mentions the information corresponding to a blank, choose the matching option (A-D).
- If the caption does NOT mention that detail and it cannot be reasonably inferred from the given information, choose E (“not given”).
- Only use general/common knowledge reasoning if it is strongly justified (for example, hearing a V10 engine can suggest a roadster).
- Unless the information is fully certain, do NOT guess.
- For each blank, output the letter (A/B/C/D/E) along with the answer.
- Output the final answer as a JSON object where keys are the blank numbers (as strings) and values are the option letters with the answer.
- Do NOT include any explanation or extra text in your output.

**Output format example:**
```json
{{
    "1": "A: xxx",
    "2": "E: xxx",
    "3": "D: xxx"
    // ... until {number}
}}
```

Now process the following input:

Cloze passage:
{cloze}

Caption:
{prediction}

Output:
""".strip()

    def __init__(self, dataset='Omni-Cloze', pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    def _metadata_frame(self, data_file):
        data = self._load_jsonl_as_dataframe(data_file)
        data['index'] = data['uuid'] if 'uuid' in data else range(len(data))
        data['video'] = data['video_path'].map(lambda p: osp.splitext(osp.normpath(str(p)))[0])
        data['question'] = data['cloze'].map(lambda c: self._parse_cloze(c).get('passage', ''))
        return data

    def _prepare_metadata(self, data_root):
        source_file = osp.join(data_root, self.DATA_FILE)
        table_file = osp.join(data_root, self.TABLE_FILE)
        required = {'index', 'video', 'video_path', 'question', 'cloze'}
        if osp.isfile(table_file) and osp.getmtime(table_file) >= osp.getmtime(source_file):
            try:
                table = load(table_file)
                if required.issubset(table.columns):
                    return table_file
            except Exception:
                pass
        self._metadata_frame(source_file).to_csv(table_file, sep='\t', index=False)
        return table_file

    @classmethod
    def supported_datasets(cls):
        return ['Omni-Cloze']

    @classmethod
    def _dataset_ready(cls, data_root):
        data_file = osp.join(data_root, cls.DATA_FILE)
        if not osp.isfile(data_file):
            return False
        try:
            records = load(data_file)
        except Exception:
            return False
        if not isinstance(records, list) or not records:
            return False
        for record in records:
            if not isinstance(record, dict) or 'video_path' not in record:
                return False
            video_path = str(record['video_path'])
            if not osp.isabs(video_path):
                video_path = osp.join(data_root, video_path)
            if not osp.isfile(osp.normpath(video_path)):
                return False
        return True

    @classmethod
    def _extract_video_archives(cls, data_root):
        archives = sorted(Path(data_root).glob(cls.VIDEO_ARCHIVE_GLOB))
        if not archives:
            raise FileNotFoundError(
                f'No {cls.VIDEO_ARCHIVE_GLOB} files found in downloaded Omni-Cloze dataset.'
            )

        root = osp.abspath(data_root)
        for archive in archives:
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    relative_path = osp.normpath(member.name).lstrip('/\\')
                    target = osp.abspath(osp.join(root, relative_path))
                    if osp.commonpath([root, target]) != root:
                        raise RuntimeError(f'Unsafe path in Omni-Cloze archive: {member.name}')
                    if member.isdir():
                        os.makedirs(target, exist_ok=True)
                    elif member.isfile():
                        os.makedirs(osp.dirname(target), exist_ok=True)
                        source = tar.extractfile(member)
                        if source is None:
                            raise RuntimeError(f'Failed to read {member.name} from {archive}')
                        with source, open(target, 'wb') as output:
                            shutil.copyfileobj(source, output)
                    else:
                        raise RuntimeError(f'Unsupported entry in Omni-Cloze archive: {member.name}')

    def prepare_dataset(self, dataset='Omni-Cloze'):
        lmu_root = LMUDataRoot()
        candidates = [osp.join(lmu_root, dataset), lmu_root]
        data_root = next((root for root in candidates if self._dataset_ready(root)), None)
        if data_root is None:
            data_root = candidates[0]
        os.makedirs(data_root, exist_ok=True)
        lock_path = osp.join(data_root, '.omni_cloze.prepare.lock')
        with portalocker.Lock(lock_path, 'w', timeout=3600):
            if self._dataset_ready(data_root):
                data_file = self._prepare_metadata(data_root)
                return dict(root=data_root, data_file=data_file)

            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=self.REPO_ID,
                repo_type='dataset',
                local_dir=data_root,
            )
            self._extract_video_archives(data_root)
            if not self._dataset_ready(data_root):
                raise RuntimeError(f'Omni-Cloze dataset preparation failed under {data_root}')
            data_file = self._prepare_metadata(data_root)
            return dict(root=data_root, data_file=data_file)

    def _load_jsonl_as_dataframe(self, data_file):
        records = load(data_file)
        if not isinstance(records, list):
            raise TypeError(f'Omni-Cloze expects JSONL records, got {type(records)} from {data_file}')
        data = pd.DataFrame(records)
        required = {'video_path', 'cloze'}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f'Omni-Cloze metadata is missing required field(s): {missing}')
        return data

    @staticmethod
    def _parse_cloze(cloze):
        if isinstance(cloze, dict):
            return copy.deepcopy(cloze)
        if isinstance(cloze, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(cloze)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        raise TypeError(f'Invalid Omni-Cloze cloze payload: {type(cloze)}')

    @staticmethod
    def _parse_prediction(raw):
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        raw = raw.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    @staticmethod
    def _parse_metadata(metadata):
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(metadata)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        return {}

    @classmethod
    def build_judge_prompt(cls, cloze, prediction, modality='audio-visual', number=None):
        cloze_data = cls._parse_cloze(cloze)
        for blank in cloze_data.get('blanks', []):
            for key in ['answer', 'distractors', 'answer_index', 'answer_key', 'required_modality']:
                blank.pop(key, None)
        if number is None:
            number = cls.DEFAULT_BLANKS
        return cls.JUDGE_PROMPT_TEMPLATE.format(
            cloze=cloze_data,
            number=number,
            prediction=prediction,
            modality=modality,
        )

    def build_prompt(self, line, video_llm=True):
        if isinstance(line, int):
            line = self.data.iloc[line]
        message = [dict(type='text', value=self.CAPTION_PROMPT)]
        if video_llm or not self.split_frame:
            message.append(dict(type='video', value=self._resolve_line_video_path(line)))
        else:
            message.extend(
                dict(type='image', value=frame_path)
                for frame_path in self.save_video_frames(line['video'])
            )
        return message

    def _resolve_line_video_path(self, line):
        video_path = str(line['video_path'])
        if not osp.isabs(video_path):
            video_path = osp.join(self.data_root, video_path)
        return osp.normpath(video_path)

    @classmethod
    def _default_cloze_prediction(cls, cloze):
        cloze_data = cls._parse_cloze(cloze)
        blanks = cloze_data.get('blanks', [])
        return {str(i): 'E: not given' for i in range(1, len(blanks) + 1)}

    @classmethod
    def _is_valid_prediction(cls, parsed, number):
        if not isinstance(parsed, dict):
            return False
        expected_keys = {str(i) for i in range(1, number + 1)}
        if set(parsed.keys()) != expected_keys:
            return False
        for value in parsed.values():
            if not isinstance(value, str) or ':' not in value:
                return False
            if value.split(':', 1)[0] not in {'A', 'B', 'C', 'D', 'E'}:
                return False
        return True

    @classmethod
    def _score_rows(cls, rows):
        total_counts = {'correct': 0, 'unknown': 0, 'wrong': 0, 'total': 0}
        subcat_stats = defaultdict(lambda: {'correct': 0, 'unknown': 0, 'wrong': 0, 'total': 0})
        modality_stats = defaultdict(lambda: {'correct': 0, 'unknown': 0, 'wrong': 0, 'total': 0})

        for item in rows:
            subcat = cls._parse_metadata(item.get('metadata')).get('subcategory', 'unknown')
            blanks = item.get('cloze', {}).get('blanks', [])
            predictions = item.get('cloze_prediction', {})
            for i, blank in enumerate(blanks, start=1):
                gold = blank.get('answer_key')
                pred_raw = predictions.get(str(i))
                if not pred_raw:
                    continue
                pred = pred_raw.split(':', 1)[0]
                modality = blank.get('required_modality', 'unknown')
                if not gold or not pred:
                    continue
                total_counts['total'] += 1
                subcat_stats[subcat]['total'] += 1
                modality_stats[modality]['total'] += 1
                if pred == gold:
                    total_counts['correct'] += 1
                    subcat_stats[subcat]['correct'] += 1
                    modality_stats[modality]['correct'] += 1
                elif pred == 'E':
                    total_counts['unknown'] += 1
                    subcat_stats[subcat]['unknown'] += 1
                    modality_stats[modality]['unknown'] += 1
                else:
                    total_counts['wrong'] += 1
                    subcat_stats[subcat]['wrong'] += 1
                    modality_stats[modality]['wrong'] += 1

        total = total_counts['total']
        score = 0.0 if total == 0 else (total_counts['correct'] - total_counts['wrong']) / total
        return {
            'correct': total_counts['correct'],
            'unknown': total_counts['unknown'],
            'wrong': total_counts['wrong'],
            'total': total,
            'score': score,
            'per_modality': dict(modality_stats),
            'per_subcategory': dict(subcat_stats),
        }

    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs = dict(judge_kwargs)
        judge_name = judge_kwargs.pop('model', None) or self.DEFAULT_JUDGE_MODEL
        nproc = judge_kwargs.pop('nproc', 4)
        number = int(judge_kwargs.pop('number', self.DEFAULT_BLANKS))
        modality = judge_kwargs.pop('modality', 'audio-visual')
        judge_kwargs.setdefault('system_prompt', self.JUDGE_SYSTEM_PROMPT)

        judge_output_file = get_intermediate_file_path(eval_file, f'_{judge_name}', 'jsonl')
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}_judge', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{judge_name}_score', 'json')

        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        indices = [line['index'] for line in lines]
        tups = [(line, modality, number, self.dataset_name) for line in lines]

        ans = load(tmp_file) if osp.exists(tmp_file) else {}
        ans = {k: v for k, v in ans.items() if self.FAIL_MSG not in str(v)}

        todo_tups = [x for x, i in zip(tups, indices) if i not in ans]
        todo_indices = [i for i in indices if i not in ans]
        if len(todo_indices):
            judge = build_judge(model=judge_name, **judge_kwargs)

            todo_tups = [(judge, *x) for x in todo_tups]
            track_progress_rich(
                OmniCloze_auxeval,
                todo_tups,
                nproc=nproc,
                chunksize=nproc,
                keys=todo_indices,
                save=tmp_file,
            )
            ans = load(tmp_file)

        output_rows = []
        for line in lines:
            item = line.to_dict()
            item.update(ans[line['index']])
            output_rows.append(item)

        dump(output_rows, judge_output_file)

        stats = self._score_rows(output_rows)
        dump(stats, score_file)
        return stats
