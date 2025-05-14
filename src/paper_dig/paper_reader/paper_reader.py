from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from paper_dig.paper import Paper
from paper_dig.summarizer import AbstractSummarizer
from paper_dig.keyword_extractor import AbstractKeywordExtractor
from paper_dig.markdown_extractor import AbstractMarkdownExtractor
from paper_dig.paper_reader.prompts import (
    extract_paper_info_prompt,
    extract_abstract_prompt,
)


class PaperReader:
    def __init__(
        self,
        llm: BaseChatModel,
        markdown_extractor: AbstractMarkdownExtractor,
        summarizer: AbstractSummarizer,
        keyword_extractor: AbstractKeywordExtractor,
    ):
        """
        Initialize the paper reader.

        Args:
            paper_path: The path to the paper.
            llm: The language model to use for paper reading.
            markdown_extractor: The markdown extractor to use for extracting markdown from the paper.
            summarizer: The summarizer to use for summarizing the paper.
            keyword_extractor: The keyword extractor to use for extracting keywords from the paper.

        Returns:
            None
        """
        self.llm = llm
        self.markdown_extractor = markdown_extractor
        self.summarizer = summarizer
        self.keyword_extractor = keyword_extractor
        self.extract_paper_info_chain = (
            extract_paper_info_prompt | self.llm | JsonOutputParser()
        )
        self.extract_abstract_chain = (
            extract_abstract_prompt | self.llm | StrOutputParser()
        )

    def read(self, paper_path: Path) -> Paper:
        """
        Read the paper.

        Args:
            paper_path: The path to the paper.

        Returns:
            Paper: The paper.
        """
        text = self.markdown_extractor.extract_markdown(paper_path)
        summary = self.summarizer.summarize(text)
        keywords = self.keyword_extractor.extract_keywords(summary)
        paper_info: dict = self.extract_paper_info_chain.invoke({"text": text[:1000]})
        abstract = self.extract_abstract_chain.invoke({"text": text[:20000]})

        return Paper(
            title=paper_info.get("title", ""),
            authors=paper_info.get("authors", []),
            year=paper_info.get("year", None),
            abstract=abstract,
            keywords=keywords,
            summary=summary,
            text=text,
        )
