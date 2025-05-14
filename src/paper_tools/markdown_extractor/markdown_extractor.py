import abc

from pathlib import Path


class AbstractMarkdownExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_markdown(self, file_path: Path) -> str:
        raise NotImplementedError
