import abc


class AbstractSummarizer(abc.ABC):
    @abc.abstractmethod
    def summarize(self, text: str) -> str:
        raise NotImplementedError
