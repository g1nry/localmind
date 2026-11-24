from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DirectorySource(BaseModel):
    root: Path
    exclude: list[str] = Field(default_factory=list)

    @field_validator('root')
    @classmethod
    def normalize_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def source_id(self) -> str:
        return sha256(str(self.root).encode()).hexdigest()

    def document_id(self, relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('Ожидается относительный путь внутри источника.')
        return sha256(f'{self.source_id}\0{path.as_posix()}'.encode()).hexdigest()


class IndexFileResult(BaseModel):
    path: str
    status: Literal['added', 'updated', 'unchanged', 'skipped', 'failed']
    document_id: str | None = None
    chunks: int = 0
    detail: str | None = None


class IndexReport(BaseModel):
    source_id: str
    root: str
    dry_run: bool
    files: list[IndexFileResult] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {status: sum(item.status == status for item in self.files)
                for status in ('added', 'updated', 'unchanged', 'skipped', 'failed')}
