"""
Split markdown:
https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger
from pydantic import BaseModel, Field


class MarkdownDocumentSplitterConfig(BaseModel):
    markdown_file_list: List[str] = Field(...)
    """markdown文件列表"""


class MarkdownDocumentSplitter:
    config: MarkdownDocumentSplitterConfig

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    def __init__(self, config: MarkdownDocumentSplitterConfig) -> None:
        self.config = config

    def split_markdown_files(self) -> List[Document]:
        """Split markdown files into smaller chunks."""
        all_splits: List[Document] = []
        for md_file in self.config.markdown_file_list:
            with open(md_file, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            splits = self._split_markdown_document(markdown_content)
            all_splits.extend(splits)
        return all_splits

    def _split_markdown_document(self, markdown_document: str) -> List[Document]:
        """Split markdown documents into smaller chunks."""
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

        md_header_splits = markdown_splitter.split_text(markdown_document)
        # for i, md_split in enumerate(md_header_splits):
        #     print(f"分割 {i}:  {repr(md_split)}\n")
        chunk_size = 500
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(md_header_splits)
        return splits


def main():
    splitter = MarkdownDocumentSplitter(
        config=MarkdownDocumentSplitterConfig(
            markdown_file_list=[
                ""
            ]
        )
    )

    docs = splitter.split_markdown_files()


if __name__ == "__main__":
    main()
