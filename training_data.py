"""TUT recipe-based source isolation and training-only event crop positions."""

from pathlib import Path, PurePosixPath
import re
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def load_recipes(directory):
    import yaml
    recipes = {}
    for path in sorted(Path(directory).rglob('mixture_recipes_*.yaml')):
        with path.open(encoding='utf-8') as source:
            rows = yaml.safe_load(source)
        if not isinstance(rows, list):
            raise ValueError(f'Expected a list of mixture recipes: {path}')
        for row in rows:
            if not isinstance(row, dict) or not all(key in row for key in ('mixture_audio_filename', 'bg_path', 'event_present')):
                raise ValueError(f'Missing required recipe fields: {path}')
            name = PurePosixPath(str(row['mixture_audio_filename']).replace('\\', '/')).name
            if name in recipes:
                raise ValueError(f'Duplicate mixture recipe: {name}')
            if not isinstance(row['event_present'], bool) or not isinstance(row['bg_path'], str):
                raise ValueError(f'Invalid recipe source fields: {name}')
            if row['event_present'] and not all(key in row for key in (
                    'event_path', 'event_class', 'event_start_in_mixture_seconds', 'event_length_seconds')):
                raise ValueError(f'Missing event source/timing fields: {name}')
            recipes[name] = row
    if not recipes:
        raise ValueError(f'No mixture_recipes_*.yaml files found in {directory}')
    return recipes


def recipe_sources(recipe):
    background = PurePosixPath(recipe['bg_path'].replace('\\', '/'))
    # Group adjacent slices such as a001_0_30.wav and a001_30_60.wav together.
    session = re.sub(r'_\d+_\d+$', '', background.stem)
    sources = {'background:' + str(background.parent / session)}
    if recipe['event_present']:
        event = PurePosixPath(str(recipe['event_path']).replace('\\', '/'))
        sources.add('event:' + str(event))  # All segments of a source recording stay together.
    return sources


def grouped_recording_split(recordings, recipes):
    sources = {}
    for subset, records in recordings.items():
        for path, event in records:
            name = Path(path).name
            if name not in recipes:
                raise ValueError(f'Missing mixture recipe for {name}')
            recipe = recipes[name]
            if recipe['event_present'] != (event is not None):
                raise ValueError(f'Recipe/CSV presence mismatch: {name}')
            if event is not None:
                start = float(recipe['event_start_in_mixture_seconds'])
                end = start + float(recipe['event_length_seconds'])
                if recipe['event_class'] != event[2] or not np.allclose([start, end], event[:2], atol=1e-5, rtol=0):
                    raise ValueError(f'Recipe/CSV event mismatch: {name}')
            sources[name] = recipe_sources(recipe)

    train_records, test_records = recordings['devtrain'], recordings['devtest']
    train_sources = set().union(*(sources[Path(path).name] for path, _ in train_records))
    test_sources = set().union(*(sources[Path(path).name] for path, _ in test_records))
    if train_sources & test_sources:
        raise ValueError('Source overlap between devtrain and devtest; check recipes and dataset provenance')

    # Connected components enforce BOTH event and background separation. This also
    # handles transitive chains (A shares background with B, B shares event with C).
    parents = list(range(len(train_records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owner = {}
    for index, (path, _) in enumerate(train_records):
        for source in sorted(sources[Path(path).name]):
            if source in owner:
                parents[root(index)] = root(owner[source])
            else:
                owner[source] = index
    groups = np.array([root(index) for index in range(len(train_records))])
    labels = np.array([int(event is not None and event[2] == 'glassbreak') for _, event in train_records])
    best = None
    if len(set(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=200, test_size=0.2, random_state=42)
        for train_ids, val_ids in splitter.split(np.zeros(len(groups)), labels, groups):
            if set(labels[train_ids]) != {0, 1} or set(labels[val_ids]) != {0, 1}:
                continue
            cost = abs(len(val_ids) / len(groups) - 0.2) + abs(labels[val_ids].mean() - labels.mean())
            if best is None or cost < best[0]:
                best = (cost, train_ids, val_ids)
    if best is not None:
        _, train_ids, val_ids = best
        group_names = {value: f'devtrain-group-{index:04d}' for index, value in enumerate(sorted(set(groups)))}
        group_map = {Path(path).name: group_names[groups[index]] for index, (path, _) in enumerate(train_records)}
    else:
        # When shared sources connect most mixtures, hold out sources and quarantine
        # connecting mixtures. No excluded audio is deleted or relabeled.
        train_ids, val_ids = source_holdout_split(train_records, sources, labels)
        group_map = {Path(path).name: 'excluded-cross-source' for path, _ in train_records}
        for name, indices in [('source-holdout-train', train_ids), ('source-holdout-validation', val_ids)]:
            for index in indices:
                group_map[Path(train_records[index][0]).name] = name
        print(f'Source holdout excludes {len(train_records) - len(train_ids) - len(val_ids)} '
              'cross-linked devtrain mixtures for this run; see split_manifest.csv.', flush=True)
    train_keys = set().union(*(sources[Path(train_records[i][0]).name] for i in train_ids))
    val_keys = set().union(*(sources[Path(train_records[i][0]).name] for i in val_ids))
    if train_keys & val_keys:
        raise AssertionError('Internal error: source leakage in grouped split')
    print(f'Source-isolated split: {len(train_ids)} train and {len(val_ids)} validation recordings', flush=True)
    return [train_records[i] for i in train_ids], [train_records[i] for i in val_ids], group_map


def source_holdout_split(records, sources, labels):
    rng = np.random.RandomState(42)
    source_sets = [sources[Path(path).name] for path, _ in records]
    all_sources = set().union(*source_sets)
    backgrounds = sorted(source for source in all_sources if source.startswith('background:'))
    events = sorted(source for source in all_sources if source.startswith('event:'))
    best = None
    for _ in range(400):
        held = set(rng.choice(backgrounds, max(1, round(len(backgrounds) * rng.uniform(0.15, 0.45))), replace=False))
        if events:
            held.update(rng.choice(events, max(1, round(len(events) * rng.uniform(0.15, 0.45))), replace=False))
        validation = [i for i, keys in enumerate(source_sets) if keys <= held]
        if not validation or set(labels[validation]) != {0, 1}:
            continue
        validation_sources = set().union(*(source_sets[i] for i in validation))
        training = [i for i, keys in enumerate(source_sets) if not keys & validation_sources]
        if not training or set(labels[training]) != {0, 1}:
            continue
        retained = len(training) + len(validation)
        cost = (abs(len(validation) / retained - 0.2)
                + abs(labels[validation].mean() - labels.mean())
                + 0.25 * (1 - retained / len(records)))
        if best is None or cost < best[0]:
            best = (cost, training, validation)
    if best is None:
        raise ValueError('All devtrain mixtures are too interconnected to form independent training/validation '
                         'with both classes. Provide more independent sources; no file-split fallback is used.')
    return best[1:]


def event_crop_starts(event_start, event_end, audio_seconds, count):
    if count < 0 or not isinstance(count, int):
        raise ValueError('Event crop count must be a nonnegative integer')
    if not count:
        return []
    if not 0 <= event_start < event_end <= audio_seconds + 1e-6:
        raise ValueError('Event timing exceeds audio duration')
    latest = max(0.0, audio_seconds - 3.0)
    if event_end - event_start <= 3:
        low, high = max(0.0, event_end - 3), min(event_start, latest)
    else:
        # An event longer than a window cannot fit in one crop; cover its portions.
        low, high = event_start, min(event_end - 3, latest)
    return np.unique(np.linspace(low, max(low, high), count)).tolist() if count > 1 else [(low + high) / 2]
