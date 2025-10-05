from pydantic import BaseModel
from typing import List, Optional


class RememberRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = None
