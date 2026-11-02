"""Phase 19.2/19.4 — trajectory corpus and Mineflayer sandbox contract."""
import random
import shutil

import pytest
import torch

from notebook.mineflayer_training_data import (GENERATORS, SKILLS, Trajectory, dataset_stats,
                                               mineflayer_trajectories)
from notebook import test_mineflayer as harness
from notebook.phone_model import Config, PhoneLM

NODE = shutil.which('node')
requires_node = pytest.mark.skipif(NODE is None, reason='node is required for the JS harness')


def tiny():
    return PhoneLM(Config(d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=128,
                          gradient_checkpointing=False))


def test_corpus_size_skills_and_determinism():
    stats = dataset_stats()
    assert stats['total'] >= 500
    assert set(stats['skills']) == set(SKILLS)
    assert all(count >= 50 for count in stats['skills'].values())
    first = [row.render() for row in mineflayer_trajectories(seed=19)]
    second = [row.render() for row in mineflayer_trajectories(seed=19)]
    assert first == second
    other = [row.render() for row in mineflayer_trajectories(seed=20)]
    assert sorted(first) == sorted(other)  # same corpus, different order
    assert first != other
    assert all(row.splitlines()[0].startswith('Request: ') for row in first)
    assert all('\nAction: ' in row for row in first)


def test_trajectory_shape_is_trainable_text():
    rows = mineflayer_trajectories()
    for row in rows[:50]:
        lines = row.render().splitlines()
        assert lines[0].startswith('Request: ')
        assert lines[1].startswith('Thought: ')
        assert lines[2].startswith('Action: ')
        assert any(line.startswith('Observation: ') for line in lines)
        assert lines[-1].startswith('Answer: ')
        assert len(row.render().encode('utf8')) < 2048
        assert row.expect['type']
        assert isinstance(row.mock, dict) and 'registry' in row.mock


def test_generators_are_independent_and_seeded():
    rows = {skill: GENERATORS[skill](random.Random(19)) for skill in SKILLS}
    assert all(len(value) >= 50 for value in rows.values())
    assert sum(len(value) for value in rows.values()) >= 500


def test_static_guard_rejects_wrong_or_unsafe_code():
    good = 'await bot.pathfinder.goto(new GoalNear(100, 64, 200, 1));'
    assert harness.static_issues(good) == []
    assert harness.syntax_check(good) in (True, None)
    bad_cases = {
        'python print': 'print("hello")',
        'exec escape': "await bot.chat('hi'); __import__('os').system('ls')",
        'require': "const fs = require('fs');",
        'process': 'process.exit(1);',
        'eval': 'eval("1 + 1");',
        'unbalanced': 'if (true) { await bot.chat("x");',
        'no bot call': 'const x = 1;',
        'python def': 'def main():\n    pass',
    }
    for label, code in bad_cases.items():
        assert harness.static_issues(code), label
    assert harness.static_warnings("bot.nearestEntity('zombie')")
    assert harness.static_warnings('bot.blockAt(50, 30, 75)')
    assert harness.static_warnings('bot.pathfinder.goto(new GoalBlock(1, 64, 1));') == []


def test_extract_action_from_free_running_text():
    text = ('Thought: I should mine.\nAction: await bot.dig(bot.blockAt(new Vec3(1, 2, 3)));\n'
            'Observation: mined\nAnswer: done')
    assert harness.extract_action(text) == 'await bot.dig(bot.blockAt(new Vec3(1, 2, 3)));'
    fenced = 'Action:\n```js\nbot.chat("hi");\n```\nObservation: ok'
    assert harness.extract_action(fenced) == 'bot.chat("hi");'
    assert harness.extract_action('no action here') == ''


def test_trajectory_validation_never_sends_static_failures_to_node(monkeypatch):
    good = Trajectory('chat', 'Say hello.', 'I should greet.', "bot.chat('hello');",
                      'The message was sent.', 'I greeted the players.', mock={},
                      expect={'type': 'chat', 'contains': 'hello'})
    bad = Trajectory('chat', 'Say hello.', 'I should greet.', 'print("hello")',
                     'The message was sent.', 'I greeted the players.', mock={},
                     expect={'type': 'chat', 'contains': 'hello'})
    submitted = []
    def fake_batch(cases):
        submitted.extend(cases)
        return [{'ok': True, 'actions': [{'type': 'chat', 'message': 'hello'}]}]
    monkeypatch.setattr(harness, 'run_mock_batch', fake_batch)
    report = harness.validate_trajectories([bad, good])
    assert [case['code'] for case in submitted] == [good.action]
    assert report['passed'] == 1 and report['static_failure_count'] == 1


def test_expectation_checker_detects_wrong_action():
    traj = next(row for row in mineflayer_trajectories()
                if row.expect['type'] == 'dig' and row.expect['block'] == 'diamond_ore')
    wrong = {'ok': True, 'actions': [{'type': 'dig', 'block': 'stone'}]}
    right = {'ok': True, 'actions': [{'type': 'dig', 'block': 'diamond_ore'}]}
    assert harness.check_expect(traj.expect, wrong)[0] is False
    assert harness.check_expect(traj.expect, right)[0] is True
    goto = {'type': 'goto', 'target': [100, 64, 200], 'within': 1}
    assert harness.check_expect(goto, {'actions': [{'type': 'goto', 'target': [100, 64, 200], 'within': 1}]})[0]
    assert not harness.check_expect(goto, {'actions': [{'type': 'goto', 'target': [0, 64, 0], 'within': 1}]})[0]


@requires_node
def test_reference_cases_pass_all_three_layers():
    rows = harness.self_test()
    assert [row['case'] for row in rows] == [case['name'] for case in harness.TEST_CASES]
    assert all(row['pass'] for row in rows), rows
    assert all(row['syntax'] for row in rows)


@requires_node
def test_entire_corpus_executes_in_the_mock():
    summary = harness.validate_trajectories()
    assert summary['sandbox'] is True
    assert summary['static_failure_count'] == 0
    assert summary['runtime_failure_count'] == 0, summary['runtime_failures']
    assert summary['passed'] == summary['total'] >= 500


@requires_node
def test_mock_rejects_invalid_calls():
    bad = harness.run_mock('await bot.dig(null);', {})
    assert bad['ok'] is False and 'null' in bad['error']
    air = harness.run_mock('await bot.placeBlock(bot.blockAt(new Vec3(0, 64, 0)), new Vec3(0, 1, 0));',
                           {'world': {'0,64,0': 'air'}, 'inventory': ['torch']})
    assert air['ok'] is False
    escaped = harness.run_mock("await bot.chat(String(process.env));", {})
    assert escaped['ok'] is False and 'process' in escaped['error']


def test_evaluate_mineflayer_returns_honest_rows():
    model = tiny()
    torch.manual_seed(0)
    rows = harness.evaluate_mineflayer(model, max_new=2, use_sandbox=False)
    assert len(rows) == len(harness.TEST_CASES)
    assert all(isinstance(row['pass'], bool) and isinstance(row['output'], str) for row in rows)
    assert all(row['action'] == '' or isinstance(row['action'], str) for row in rows)


def test_harness_never_uses_python_exec_or_eval():
    source = (harness.__file__ and open(harness.__file__, encoding='utf8').read()) or ''
    assert 'exec(model_output)' not in source
    assert 'subprocess.run' in source
    js = open(harness.HARNESS, encoding='utf8').read()
    assert 'vm.createContext' in js and 'codeGeneration' in js
    assert 'require(' not in js.replace("require('node:vm')", '')
