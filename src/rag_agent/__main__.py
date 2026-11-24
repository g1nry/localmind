import argparse
import asyncio
import json
import sys
from pathlib import Path

from rag_agent.bootstrap import open_services
from rag_agent.core.config import get_settings
from rag_agent.domain.sources import DirectorySource
from rag_agent.storage.errors import StorageError


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
    args = parser.parse_args()

    try:
        settings = get_settings()
        if args.command == 'index':
            if args.reload:
                parser.error('--reload применяется только к backend.')
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
    except (OSError, ValueError, StorageError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)

    import uvicorn
    uvicorn.run('rag_agent.main:app', host=settings.app_host, port=settings.app_port, reload=args.reload)


if __name__ == '__main__':
    main()
