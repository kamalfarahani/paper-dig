from langchain_core.prompts import PromptTemplate


IS_THERE_ABSTRACT_PROMPT_TEXT = """\
Given the following text check if there is an article abstract or summary in it.
If there is an abstract return true else return false.

Text:
{text}

Only output the boolean value in pure string format.
DO NOT output any other text like "Here is the boolean value".
"""


EXTRACT_PAPER_INFO_PROMPT_TEXT = """\
Extract the following information from the academic paper text into JSON format:

1. Title: The full title of the paper (typically at the beginning)
2. Authors: All authors' names (typically listed after the title)
3. Year: The publication year (typically found in the header, footer, or citation information)

JSON Schema:
{{
    "title": str,  # The complete paper title
    "authors": list[str],  # List of author names, each as "First Last" format
    "year": int,  # Publication year as a 4-digit number
}}

If any information cannot be found in the text, use `null` for that field.

Text:
{text}

Return ONLY the JSON object as a plain text string without any formatting, code blocks, markdown, or additional explanatory text.
"""

EXTRACT_ABSTRACT_PROMPT_TEXT = """\
Extract the academic paper's abstract from the following text. The abstract is typically:

1. Found near the beginning of the paper, after the title and authors
2. Often labeled with the heading "Abstract" or "Summary"
3. A concise summary of the paper's purpose, methods, results, and conclusions
4. Usually a single paragraph or a few paragraphs of dense text

Requirements:
- The abstract should be between 200-500 words
- If the original abstract is shorter than 200 words, expand it slightly based on information from the paper
- If the original abstract is longer than 500 words, condense it while preserving key information
- Maintain the academic tone and technical terminology of the original

Text:
{text}

Return ONLY the abstract text as a plain string without any formatting, quotation marks, headers, or explanatory text.
"""

is_there_abstract_prompt = PromptTemplate.from_template(
    IS_THERE_ABSTRACT_PROMPT_TEXT,
)

extract_paper_info_prompt = PromptTemplate.from_template(
    EXTRACT_PAPER_INFO_PROMPT_TEXT,
)

extract_abstract_prompt = PromptTemplate.from_template(
    EXTRACT_ABSTRACT_PROMPT_TEXT,
)
