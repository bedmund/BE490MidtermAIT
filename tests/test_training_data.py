import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from training_data import event_crop_starts, grouped_recording_split, load_recipes, recipe_sources, ValidationPolicy, coverage_rows, coverage_counts


class TrainingDataTests(unittest.TestCase):
    # Tiny fixtures explicitly lower the production coverage floor.
    policy = ValidationPolicy(2, 2, 2, 2, 1)

    def data(self):
        records = {'devtrain': [], 'devtest': []}
        recipes = {}
        for subset, groups in [('devtrain', 10), ('devtest', 2)]:
            for group in range(groups):
                for variant in range(3):
                    name = f'mixture_{subset}_glassbreak_{group}_{variant}.wav'
                    event = (8.9, 9.5, 'glassbreak') if variant < 2 else None
                    records[subset].append((name, event))
                    recipes[name] = dict(mixture_audio_filename=name, bg_path=f'audio/{subset}-{group}_0_30.wav',
                                         event_present=event is not None, event_path=f'glassbreak/{subset}-{group}.wav',
                                         event_class='glassbreak', event_start_in_mixture_seconds=8.9,
                                         event_length_seconds=0.6)
        return records, recipes

    def test_shared_sources_never_cross_splits_and_result_is_repeatable(self):
        records, recipes = self.data()
        with contextlib.redirect_stdout(io.StringIO()):
            train, val, groups = grouped_recording_split(records, recipes, policy=self.policy)
            again = grouped_recording_split(records, recipes, policy=self.policy)
        self.assertEqual(train, again[0])
        self.assertEqual(val, again[1])
        source_sets = []
        for split in [train, val, records['devtest']]:
            source_sets.append(set().union(*(recipe_sources(recipes[Path(path).name]) for path, _ in split)))
        for i in range(3):
            for j in range(i):
                self.assertFalse(source_sets[i] & source_sets[j])
        self.assertEqual(len(train) + len(val), len(records['devtrain']))
        self.assertEqual(len(set(groups.values())), 10)

    def test_transitive_sources_and_adjacent_background_slices_group_together(self):
        records, recipes = self.data()
        names = [path for path, _ in records['devtrain']]
        # Different groups now share an event source; their backgrounds become linked too.
        recipes[names[3]]['event_path'] = recipes[names[0]]['event_path']
        recipes[names[4]]['bg_path'] = recipes[names[3]]['bg_path'].replace('_0_30', '_30_60')
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, groups = grouped_recording_split(records, recipes, policy=self.policy)
        self.assertEqual(groups[names[0]], groups[names[3]])
        self.assertEqual(groups[names[3]], groups[names[4]])

    def test_missing_conflicting_and_connected_recipes_fail_explicitly(self):
        records, recipes = self.data()
        missing = dict(recipes)
        missing.pop(records['devtrain'][0][0])
        with self.assertRaisesRegex(ValueError, 'Missing mixture recipe'):
            grouped_recording_split(records, missing, policy=self.policy)
        recipes[records['devtrain'][0][0]]['event_present'] = False
        with self.assertRaisesRegex(ValueError, 'presence mismatch'):
            grouped_recording_split(records, recipes, policy=self.policy)
        records, recipes = self.data()
        for path, _ in records['devtrain']:
            recipes[path]['bg_path'] = 'audio/shared_0_30.wav'
        with self.assertRaisesRegex(ValueError, 'Insufficient independent|All devtrain mixtures'):
            grouped_recording_split(records, recipes, policy=self.policy)

    def test_production_policy_rejects_tiny_validation_collection_and_reports_counts(self):
        records, recipes = self.data()
        diagnostics = {}
        with self.assertRaisesRegex(ValueError, 'Insufficient independent devtrain coverage'):
            grouped_recording_split(records, recipes, diagnostics=diagnostics)
        report = diagnostics['coverage']
        self.assertEqual(report['status'], 'insufficient_available_coverage')
        self.assertEqual(report['available_devtrain']['glass_recordings'], 20)
        self.assertEqual(report['available_devtrain']['glass_events'], 10)
        self.assertEqual(report['policy']['min_glass_recordings'], 20)

    def test_repeated_mixtures_do_not_inflate_distinct_glass_events(self):
        records, recipes = self.data()
        for path, event in records['devtrain']:
            if event is not None:
                recipes[path]['event_path'] = 'glassbreak/shared.wav'
                recipes[path]['segment_start_seconds'] = 1.0
                recipes[path]['segment_end_seconds'] = 1.6
        rows = coverage_rows(records['devtrain'], recipes)
        counts = coverage_counts(rows, range(len(rows)))
        self.assertEqual(counts['glass_recordings'], 20)
        self.assertEqual(counts['glass_events'], 1)
        self.assertEqual(counts['glass_sources'], 1)

    def test_accepted_split_meets_explicit_coverage_policy(self):
        records, recipes = self.data()
        diagnostics = {}
        with contextlib.redirect_stdout(io.StringIO()):
            grouped_recording_split(records, recipes, policy=self.policy, diagnostics=diagnostics)
        report = diagnostics['coverage']
        self.assertEqual(report['status'], 'accepted')
        for split in ('train', 'validation'):
            for key, minimum in report['policy'].items():
                self.assertGreaterEqual(report[split][key.removeprefix('min_')], minimum)

    def test_yaml_recipe_format_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No mixture_recipes'):
                load_recipes(directory)
            Path(directory, 'mixture_recipes_devtrain_glassbreak.yaml').write_text(
                '- mixture_audio_filename: mixture_devtrain_glassbreak_0.wav\n'
                '  bg_path: audio/a001_0_30.wav\n  event_present: false\n')
            self.assertIn('mixture_devtrain_glassbreak_0.wav', load_recipes(directory))

    def test_connected_sources_use_explicit_exclusions_without_leakage(self):
        records, recipes = self.data()
        names = [path for path, _ in records['devtrain']]
        for group in range(1, 10):
            recipes[names[group * 3]]['event_path'] = recipes[names[(group - 1) * 3 + 1]]['event_path']
        with contextlib.redirect_stdout(io.StringIO()) as output:
            train, val, groups = grouped_recording_split(records, recipes, policy=self.policy)
        self.assertLess(len(train) + len(val), len(names))
        train_keys = set().union(*(recipe_sources(recipes[path]) for path, _ in train))
        val_keys = set().union(*(recipe_sources(recipes[path]) for path, _ in val))
        self.assertFalse(train_keys & val_keys)
        self.assertIn('excludes', output.getvalue())
        self.assertIn('excluded-cross-source', groups.values())

    def test_event_crops_contain_short_events_and_cover_long_ones(self):
        starts = event_crop_starts(8.9, 9.5, 30, 3)
        self.assertEqual(len(starts), 3)
        for start in starts:
            self.assertLessEqual(start, 8.9)
            self.assertGreaterEqual(start + 3, 9.5)
        self.assertEqual(event_crop_starts(0.1, 0.5, 30, 3)[0], 0)
        long = event_crop_starts(11.2, 15.8, 30, 3)
        self.assertAlmostEqual(long[0], 11.2)
        self.assertAlmostEqual(long[-1] + 3, 15.8)
        self.assertEqual(event_crop_starts(0, 0.5, 30, 0), [])
        with self.assertRaises(ValueError):
            event_crop_starts(2, 4, 3, 3)
