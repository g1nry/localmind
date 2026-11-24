import json
import os
import subprocess
import sys

import pytest

from rag_agent.__main__ import main
from rag_agent.core.config import Settings


def test_dry_run_cli_from_other_directory_does_not_write(tmp_path):
    source = tmp_path / 'knowledge'
    source.mkdir()
    (source / 'notes.md').write_text('known fact')
    config = tmp_path / 'localmind.toml'
    config.write_text('[data]\ndir = "./data"\n[sources.docs]\nroot = "./knowledge"\n')
    result = subprocess.run(
        [sys.executable, '-m', 'rag_agent', 'index', '--source', 'docs', '--dry-run', '--json'],
        cwd=tmp_path, env={**os.environ, 'RAG_AGENT_CONFIG': str(config),
                          'OLLAMA_BASE_URL': 'http://127.0.0.1:1'}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['dry_run'] and report['counts']['added'] == 1
    assert not (tmp_path / 'data').exists()


def test_cli_index_and_partial_failure_exit_code(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'good.md').write_text('known fact')
    (root / 'bad.txt').write_bytes(b'\xff')
    settings = Settings(DATA_DIR=tmp_path / 'data')
    monkeypatch.setattr('rag_agent.__main__.get_settings', lambda: settings)

    async def embed(self, model, text):
        return [1.0, 0.0]

    monkeypatch.setattr('rag_agent.integrations.ollama.client.OllamaClient.embed', embed)
    monkeypatch.setattr(sys, 'argv', ['localmind', 'index', str(root), '--json'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report['counts']['added'] == 1
    assert report['counts']['failed'] == 1
    assert settings.index_path.exists()
