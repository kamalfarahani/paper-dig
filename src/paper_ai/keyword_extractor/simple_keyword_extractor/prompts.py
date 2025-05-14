from langchain_core.prompts import ChatPromptTemplate


extract_keywords_prompt = ChatPromptTemplate.from_template(
    """
    Extract scientific and technical keywords from the following text according to these guidelines:
    
    1. Focus on domain-specific terminology, technical concepts, scientific methods, and specialized vocabulary
    2. Prioritize nouns and noun phrases that represent concrete technical concepts
    3. Include relevant acronyms and their expansions when present
    4. Extract compound technical terms as single keywords (e.g., "machine learning" rather than "machine" and "learning" separately)
    5. Exclude the following:
       - Common words and stop words (e.g., the, is, at, which, on)
       - Generic academic terms (e.g., study, research, analysis, result)
       - Non-technical adjectives and adverbs
       - Numbers unless they are part of a specific technical term
    
    Return the keywords in the following JSON schema:
    {{
        "keywords": list[str]  # List of unique, normalized keywords in lowercase
    }}
    
    Text to analyze:
    {text}
    
    Return ONLY the JSON object as a plain text string without any formatting, code blocks, markdown, or additional explanatory text.
    """
)
