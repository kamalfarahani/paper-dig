from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from paper_tools.paper import Paper

from paper_tools.pdf_to_markdown import pdf_to_markdown
from paper_tools.prompts import extract_paper_info_prompt, extract_abstract_prompt


class PaperReader:
    def __init__(
        self,
        paper_path: Path,
        llm: BaseChatModel,
        datalab_api_key: str,
    ):
        self.llm = llm
        self.paper_path = paper_path
        self.datalab_api_key = datalab_api_key
        self.extract_paper_info_chain = (
            extract_paper_info_prompt | self.llm | JsonOutputParser()
        )
        self.extract_abstract_chain = (
            extract_abstract_prompt | self.llm | StrOutputParser()
        )

    def read(self) -> Paper:
        text = pdf_to_markdown(self.paper_path, self.datalab_api_key)

        paper_info = self.extract_paper_info_chain.invoke({"text": text[:1000]})
        abstract = self.extract_abstract_chain.invoke({"text": text[:20000]})

        return Paper(
            title=paper_info["title"],
            authors=paper_info["authors"],
            year=paper_info["year"],
            abstract=abstract,
            keywords=[],
            summary="",
            text=text,
        )
