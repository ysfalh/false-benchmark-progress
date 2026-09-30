"""The report follows public feedback and exports evaluator-approved aggregates."""

import json
import numpy as np
import pytest
from scripts.maintainer_report import main


def test_report_choice_and_loss_follow_public_transcript(tmp_path):
    output = tmp_path/'report'
    report = main('examples.custom_benchmark.run:SyntheticBenchmark', 'category', 8, output)
    transcript = json.loads((output/'public_transcript.json').read_text())
    genuine = np.array(transcript['public_profiles']).mean(axis=1)
    final = np.array(transcript['candidate_and_final_feedback'][-1]).mean()
    leader = int(np.argmin(np.round(1-genuine, 10)))
    choose_router = round(1-final, 10) < round(1-genuine[leader], 10)
    assert report['no_attack_model'] == transcript['models'][leader]
    assert report['selection']['model'] == ('Final router' if choose_router else report['no_attack_model'])
    expected = report['no_attack']['heldout_score_pp']-report['attack']['heldout_score_pp'] if choose_router else 0.
    assert report['selection']['model_selection_loss_pp'] == pytest.approx(expected)
    assert len(transcript['candidate_and_final_feedback']) == 8
    assert 'private_item_scores' not in json.dumps(report)
    assert 'Model-selection loss' in (output/'report.html').read_text()
    assert (output/'style.css').is_file() and (output/'fonts/texgyrepagella-regular.woff').is_file()
    with pytest.raises(FileExistsError):
        main('examples.custom_benchmark.run:SyntheticBenchmark', 'category', 8, output)
