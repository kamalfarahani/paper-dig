from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser


class Summarizer:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _split_text(self, text: str) -> list[str]: ...

    def summarize(self, text: str) -> str: ...
