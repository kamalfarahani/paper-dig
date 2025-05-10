from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from paper_tools.paper import Paper
from paper_tools.summarizer import AbstractSummarizer
from paper_tools.keyword_extractor import AbstractKeywordExtractor
from paper_tools.markdown_extractor import AbstractMarkdownExtractor

from paper_tools.paper_reader.prompts import (
    extract_paper_info_prompt,
    extract_abstract_prompt,
)


class PaperReader:
    def __init__(
        self,
        paper_path: Path,
        llm: BaseChatModel,
        markdown_extractor: AbstractMarkdownExtractor,
        summarizer: AbstractSummarizer,
        keyword_extractor: AbstractKeywordExtractor,
    ):
        self.llm = llm
        self.paper_path = paper_path
        self.markdown_extractor = markdown_extractor
        self.summarizer = summarizer
        self.keyword_extractor = keyword_extractor
        self.extract_paper_info_chain = (
            extract_paper_info_prompt | self.llm | JsonOutputParser()
        )
        self.extract_abstract_chain = (
            extract_abstract_prompt | self.llm | StrOutputParser()
        )

    def read(self) -> Paper:
        text = self.markdown_extractor.extract_markdown(self.paper_path)
        keywords = self.keyword_extractor.extract_keywords(text)
        summary = self.summarizer.summarize(text)
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
