import numpy as np

from sklearn.cluster import KMeans

from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_experimental.text_splitter import SemanticChunker

from .prompts import summarize_prompt, summarize_aggregate_prompt


def get_most_important_documents(
    embeddings: np.ndarray,
    documents: list[str],
    num_clusters: int = 5,
) -> list[str]:
    """
    Clusters document embeddings using KMeans and returns the documents closest to the centroids.

    Args:
        embeddings (np.ndarray): A 2D numpy array where each row represents the embedding of a document.
        documents (list[str]): A list of strings, where each string is the content of a document.
        num_clusters (int, optional): The number of clusters to form. Defaults to 5.

    Returns:
        list[str]: A list of the documents that are closest to the centroids of the clusters.
                   Returns an empty list if embeddings or documents are invalid.
    """
    # Perform KMeans clustering
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
    )
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Find the document closest to each centroid
    closest_docs = []
    for centroid_index in range(num_clusters):
        # Get all documents in the current cluster.
        cluster_points = embeddings[labels == centroid_index]
        if len(cluster_points) == 0:
            # Handle the edge case where a cluster is empty.
            closest_docs.append("No documents in this cluster")
            continue

        # Calculate distances to the centroid.
        distances = np.linalg.norm(cluster_points - centroids[centroid_index], axis=1)

        # Find the index of the closest document in *this cluster*.
        closest_index_in_cluster = np.argmin(distances)

        # Need to find the global index of the closest document.
        global_index = np.where(labels == centroid_index)[0][closest_index_in_cluster]
        closest_docs.append(documents[global_index])

    return closest_docs


class SemanticSummarizer:
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
