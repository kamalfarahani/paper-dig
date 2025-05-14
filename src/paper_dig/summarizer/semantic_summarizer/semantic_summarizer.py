from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_experimental.text_splitter import SemanticChunker

from paper_dig.summarizer import AbstractSummarizer

from .prompts import summarize_prompt, summarize_aggregate_prompt


class SemanticSummarizer(AbstractSummarizer):
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
    ):
        """
        Initialize the semantic summarizer.

        Args:
            llm: The language model to use for summarization.
            embeddings: The embeddings to use for semantic chunking.

        Returns:
            None
        """
        self.llm = llm
        self.embeddings = embeddings
        self.text_splitter = SemanticChunker(embeddings=embeddings)
        self.summarize_chain = summarize_prompt | llm | StrOutputParser()
        self.summarize_aggregate_chain = (
            summarize_aggregate_prompt | llm | StrOutputParser()
        )

    def _split_text(self, text: str) -> list[str]:
        """
        Splits the text into chunks using semantic chunking.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        docs = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    def summarize(self, text: str) -> str:
        """
        Summarizes the text using semantic chunking and semantic summarization.

        Args:
            text: The text to summarize.

        Returns:
            A string containing the summary of the text.
        """
        chunks = self._split_text(text)
        summaries = [
            self.summarize_chain.invoke(
                {
                    "text": chunk,
                }
            )
            for chunk in chunks
        ]

        summaries = "\n --------------- \n ".join(summaries)

        return self.summarize_aggregate_chain.invoke(
            {
                "summaries": summaries,
            }
        )
