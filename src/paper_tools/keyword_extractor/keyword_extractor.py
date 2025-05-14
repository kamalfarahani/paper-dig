import abc


class AbstractKeywordExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_keywords(self, text: str) -> list[str]:
        raise NotImplementedError
