from typing import Optional

from pydantic import BaseModel


class Paper(BaseModel):
    title: str
    year: Optional[int]
    authors: list[str]
    keywords: list[str]
    abstract: str
    summary: str
    text: str
