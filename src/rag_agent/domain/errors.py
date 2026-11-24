class ProviderError(RuntimeError):
    """Модель или её HTTP-провайдер не смогли выполнить запрос."""


class AgentError(RuntimeError):
    """Агент не смог сформировать ответ."""
