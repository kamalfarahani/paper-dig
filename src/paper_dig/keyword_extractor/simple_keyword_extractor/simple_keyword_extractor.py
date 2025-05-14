from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

from paper_dig.keyword_extractor import AbstractKeywordExtractor

from .prompts import extract_keywords_prompt


class SimpleKeywordExtractor(AbstractKeywordExtractor):
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the simple keyword extractor.

        Args:
            llm: The language model to use for keyword extraction.

        Returns:
            None
        """
        self.llm = llm
        self.extract_keywords_chain = (
            extract_keywords_prompt | self.llm | JsonOutputParser()
        )

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract keywords from the given text.

        Args:
            text: The text to extract keywords from.

        Returns:
            A list of keywords extracted from the text.
        """
        return self.extract_keywords_chain.invoke(
            {
                "text": text,
            }
        )["keywords"]
