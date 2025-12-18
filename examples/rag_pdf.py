import os
from typing import Any, Protocol, overload

from langchain_community.document_loaders import PyPDFLoader
from loguru import logger


class Logger(Protocol):
    @overload
    def info(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805
    @overload
    def info(__self, __message: Any) -> None: ...  # noqa: N805
    @overload
    def error(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805
    @overload
    def error(__self, __message: Any) -> None: ...  # noqa: N805


def main(log: Logger):
    file_path = os.path.expandvars("$HOME/Downloads/nke-10k-2023.pdf")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    log.info(f"Number of documents: {len(docs)}", len(docs))
    log.info(f"First 100 chars: {docs[0].page_content[:100]}")
    log.info(f"Metadata: {docs[0].metadata}")


if __name__ == "__main__":
    main(logger)
    logger.info
