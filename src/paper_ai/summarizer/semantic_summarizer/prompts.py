from langchain.prompts import PromptTemplate


summarize_prompt = PromptTemplate.from_template(
    """
Summarize the following passage with precision and depth. Your summary should:

1. Be comprehensive and include at least 2 well-developed paragraphs
2. Capture key concepts, arguments, characters, events, and themes
3. Preserve the original tone and perspective of the text
4. Begin directly with the content (avoid phrases like "In this passage" or "This text discusses")
5. Present information in a logical flow that mirrors the original structure
6. Maintain appropriate context and relationships between ideas
7. Use clear, concise language while retaining important details and nuances
8. Ensure the summary could stand alone as a coherent representation of the original

Passage:

```{text}```

    SUMMARY:
"""
)

summarize_aggregate_prompt = PromptTemplate.from_template(
    """
    Aggregate the following summaries into a single coherent summary. Your aggregated summary should:
    
    1. Synthesize all key information from the individual summaries without redundancy
    2. Maintain a logical flow and structure that connects ideas across summaries
    3. Identify and emphasize recurring themes, concepts, and arguments
    4. Resolve any contradictions or inconsistencies between summaries
    5. Preserve the depth and nuance of the original summaries
    6. Be comprehensive and include at least 2-3 well-developed paragraphs
    7. Present a balanced representation of all source summaries
    8. Use clear, concise language while retaining important details
    9. Stand alone as a coherent representation of all the original content
    
    Summaries:
    ```{summaries}```
    
    AGGREGATE SUMMARY:
    """
)
