from typing import Any, Protocol, overload


class Logger(Protocol):
    @overload
    def debug(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805

    @overload
    def debug(__self, __message: Any) -> None: ...  # noqa: N805

    @overload
    def info(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805

    @overload
    def info(__self, __message: Any) -> None: ...  # noqa: N805

    @overload
    def error(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805

    @overload
    def error(__self, __message: Any) -> None: ...  # noqa: N805
