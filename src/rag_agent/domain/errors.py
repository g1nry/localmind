class ProviderError(RuntimeError):
    """Модель или её HTTP-провайдер не смогли выполнить запрос."""


class AgentError(RuntimeError):
    """Агент не смог сформировать ответ."""


class ChunkNotFoundError(LookupError):
    """Фрагмент не найден в индексе или выбранном источнике."""


class StaleChunkError(RuntimeError):
    """Версия фрагмента изменилась после поиска."""
