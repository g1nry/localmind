import fnmatch
import hashlib
import json
import os
from pathlib import Path
import stat

from rag_agent.core.config import Settings
from rag_agent.domain.contracts import VectorStore
from rag_agent.domain.errors import ProviderError
from rag_agent.domain.sources import DirectorySource, IndexFileResult, IndexReport
from rag_agent.services.document_errors import DocumentIngestionError, DocumentTooLargeError
from rag_agent.services.document_ingestion_service import DocumentIngestionService
from rag_agent.storage.errors import IndexWriteError


DEFAULT_EXCLUDES = ('.git', '.venv', '__pycache__', 'node_modules', '.env', '.env.*')


def _matches(relative: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        alternatives = [pattern]
        if pattern.startswith('**/'):
            alternatives.append(pattern[3:])
        if any(fnmatch.fnmatchcase(relative, item) for item in alternatives):
            return True
    return False


class DirectoryIndexingService:
    def __init__(self, settings: Settings, store: VectorStore, ingestion: DocumentIngestionService):
        self._settings = settings
        self._store = store
        self._ingestion = ingestion

    def _read(self, path: Path, root: Path) -> bytes:
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Файл находится вне корня источника.')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, 'rb') as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Поддерживаются только обычные файлы.')
            limit = self._settings.max_upload_size_bytes
            if before.st_size > limit:
                raise DocumentTooLargeError(limit)
            content = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise ValueError('Файл изменился во время чтения; повторите индексацию.')
        return content

    def _fingerprint(self, content: bytes) -> str:
        recipe = json.dumps({
            'chunker': 'characters-v1',
            'size': self._settings.max_chunk_size,
            'overlap': self._settings.chunk_overlap,
            'embedding_model': self._settings.ollama_embedding_model,
        }, sort_keys=True).encode()
        return hashlib.sha256(recipe + b'\0' + content).hexdigest()

    async def index(self, source: DirectorySource, *, dry_run: bool = False, force: bool = False) -> IndexReport:
        root = source.root
        if not root.is_dir():
            raise ValueError(f'Каталог источника не найден: {root}')
        reserved = [self._settings.documents_dir.resolve(), self._settings.index_path.parent.resolve()]
        if any(root.is_relative_to(path) for path in reserved):
            raise ValueError('Служебный каталог LocalMind нельзя использовать как источник.')
        report = IndexReport(source_id=source.source_id, root=str(root), dry_run=dry_run)
        existing: dict[str, list] = {}
        for record in self._store.load():
            if record.metadata.get('source_id') == source.source_id:
                existing.setdefault(record.metadata['document_id'], []).append(record)

        def scan_error(exc: OSError):
            path = Path(exc.filename) if exc.filename else root
            relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else '.'
            report.files.append(IndexFileResult(path=relative, status='failed', detail=str(exc)))

        for directory, directories, filenames in os.walk(root, topdown=True, followlinks=False, onerror=scan_error):
            directory_path = Path(directory)
            for name in sorted(directories + filenames):
                path = directory_path / name
                relative = path.relative_to(root).as_posix()
                is_directory = name in directories
                excluded = (
                    _matches(name, list(DEFAULT_EXCLUDES))
                    or _matches(relative, source.exclude)
                    or (is_directory and _matches(relative + '/__entry__', source.exclude))
                    or any(path == reserved_path for reserved_path in reserved)
                    or path.is_symlink()
                )
                if excluded:
                    if is_directory:
                        directories.remove(name)
                    report.files.append(IndexFileResult(path=relative, status='skipped', detail='Исключение или symlink.'))
                    continue
                if is_directory:
                    continue
                if path.suffix.lower() not in self._settings.allowed_document_extensions:
                    report.files.append(IndexFileResult(path=relative, status='skipped', detail='Неподдерживаемое расширение.'))
                    continue
                document_id = source.document_id(relative)
                try:
                    content = self._read(path, root)
                    self._ingestion.validate_document(relative, content)
                    fingerprint = self._fingerprint(content)
                    previous = existing.get(document_id, [])
                    if not force and previous and all(item.metadata.get('fingerprint') == fingerprint for item in previous):
                        report.files.append(IndexFileResult(path=relative, status='unchanged', document_id=document_id,
                                                           chunks=len(previous)))
                        continue
                    status = 'updated' if previous else 'added'
                    chunks = 0
                    if not dry_run:
                        chunks = await self._ingestion.ingest_source_document(source, relative, content, fingerprint)
                    report.files.append(IndexFileResult(path=relative, status=status, document_id=document_id, chunks=chunks))
                except (OSError, ValueError, DocumentIngestionError, ProviderError, IndexWriteError) as exc:
                    report.files.append(IndexFileResult(path=relative, status='failed', document_id=document_id, detail=str(exc)))
            directories.sort()
        return report
