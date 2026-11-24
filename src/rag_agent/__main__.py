import argparse
import asyncio
import json
import sys
from pathlib import Path

from rag_agent.bootstrap import open_services
from rag_agent.core.config import get_settings
from rag_agent.domain.sources import DirectorySource
from rag_agent.domain.errors import ProviderError, ChunkNotFoundError, StaleChunkError
from rag_agent.domain.schemas import FetchRequest
from rag_agent.services.fetch_service import FetchService
from rag_agent.storage.vector_store import JsonVectorStore
from rag_agent.storage.errors import StorageError


def _integer_in_range(maximum):
    def parse(value):
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError('Ожидается целое число.') from exc
        if not 1 <= number <= maximum:
            raise argparse.ArgumentTypeError(f'Ожидается число от 1 до {maximum}.')
        return number
    return parse


def _print_context(context, index=None):
    heading = f'[{index}] ' if index is not None else ''
    score = f' (score={context.score:.4f})' if context.score is not None else ''
    print(f'{heading}{context.source}{score}')
    print(f'chunk_id: {context.chunk_id}\nversion: {context.version}')
    print(context.text)
    if context.truncated:
        print('[Текст усечён по лимиту символов.]')


def main() -> None:
    parser = argparse.ArgumentParser(description='LocalMind: локальная база знаний и агент.')
    parser.add_argument('--reload', action='store_true', help='Перезапуск backend при изменениях кода.')
    commands = parser.add_subparsers(dest='command')
    index = commands.add_parser('index', help='Индексировать каталог с TXT/Markdown.')
    index.add_argument('path', nargs='?', type=Path, help='Корень каталога.')
    index.add_argument('--source', help='Имя источника из config.toml.')
    index.add_argument('--exclude', action='append', default=[], help='Дополнительная маска исключения.')
    index.add_argument('--dry-run', action='store_true', help='Показать план без записи и запросов к модели.')
    index.add_argument('--force', action='store_true', help='Пересчитать даже неизменившиеся файлы.')
    index.add_argument('--json', action='store_true', help='Вывести машиночитаемый отчёт.')
    search = commands.add_parser('search', help='Найти фрагменты без chat-модели.')
    search.add_argument('query', help='Поисковый запрос.')
    search.add_argument('--top-k', type=_integer_in_range(100), help='Максимальное число фрагментов (1–100).')
    fetch = commands.add_parser('fetch', help='Прочитать фрагмент из индекса без Ollama.')
    fetch.add_argument('chunk_id', help='Точный chunk_id из выдачи search.')
    fetch.add_argument('--version', help='Ожидаемая версия из выдачи search.')
    for command in (search, fetch):
        command.add_argument('--source', help='Ограничить именованным источником из config.toml.')
        command.add_argument('--max-chars', type=_integer_in_range(100_000), default=4000,
                             help='Лимит символов текста (по умолчанию 4000, максимум 100000).')
        command.add_argument('--json', action='store_true', help='Вывести структурированный результат.')
    args = parser.parse_args()
    if args.command and args.reload:
        parser.error('--reload применяется только к backend.')
    if args.command == 'search' and not args.query.strip():
        parser.error('Поисковый запрос не должен быть пустым.')

    try:
        settings = get_settings()
        if args.command in {'search', 'fetch'}:
            source_id = None
            if args.source:
                if args.source not in settings.sources:
                    parser.error(f'Неизвестный источник: {args.source}')
                source_id = settings.sources[args.source].source_id
            if args.command == 'fetch':
                service = FetchService(JsonVectorStore(settings.index_path))
                context = service.fetch(FetchRequest(
                    chunk_id=args.chunk_id, version=args.version,
                    source_id=source_id, max_chars=args.max_chars,
                ))
                if args.json:
                    print(json.dumps(context.model_dump(exclude_none=True), ensure_ascii=False))
                else:
                    _print_context(context)
            else:
                async def search_records():
                    async with open_services(settings) as services:
                        return await services.retrieval.retrieve_context(
                            args.query, args.top_k, source_id=source_id, max_chars=args.max_chars
                        )
                result = asyncio.run(search_records())
                if args.json:
                    print(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False))
                else:
                    if not result.contexts:
                        print('Подходящие фрагменты не найдены.')
                    for number, context in enumerate(result.contexts, 1):
                        _print_context(context, number)
                    if result.truncated:
                        print('[Выдача ограничена лимитом символов.]')
            return
        if args.command == 'index':
            if (args.path is None) == (args.source is None):
                parser.error('Укажите каталог или --source, но не оба одновременно.')
            if args.source:
                if args.source not in settings.sources:
                    parser.error(f'Неизвестный источник: {args.source}')
                source = settings.sources[args.source]
            else:
                source = DirectorySource(root=args.path)
            source = source.model_copy(update={'exclude': [*source.exclude, *args.exclude]})

            async def run():
                async with open_services(settings) as services:
                    return await services.indexing.index(source, dry_run=args.dry_run, force=args.force)

            report = asyncio.run(run())
            if args.json:
                print(json.dumps({**report.model_dump(), 'counts': report.counts}, ensure_ascii=False))
            else:
                print('План индексации:' if report.dry_run else 'Результат индексации:')
                for item in report.files:
                    print(f'{item.status}: {item.path}' + (f' — {item.detail}' if item.detail else ''))
                print(', '.join(f'{key}={value}' for key, value in report.counts.items()))
            raise SystemExit(1 if report.counts['failed'] else 0)
    except (OSError, ValueError, StorageError, ProviderError, ChunkNotFoundError, StaleChunkError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)

    import uvicorn
    uvicorn.run('rag_agent.main:app', host=settings.app_host, port=settings.app_port, reload=args.reload)


if __name__ == '__main__':
    main()
