
from typing import Dict

from pydantic import BaseModel, Field

class SummaryInput(BaseModel):
    topic: str = Field(
        description="what  is the topic of the post"
    )
    number_of_paragraphs: int = Field(
        default=2,
        ge=1,
        le=10,
        description="How many paragraphs are needed"
    )
    number_of_images: int = Field(
        default=2,
        description="max number of images to generate"
    )
    tone: str = Field(
        default="neutral",
        description="Desired tone of content such as neutral, professional, simple, or friendly"
    )
    seo_keywords: str = Field(
        default="",
        description="SEO keywords related to the topic"
    )
class SummaryOutput(BaseModel):
    content: str = Field(description="Full content with image_1, image_2 placeholders")
    title:str= Field(description="Title for the content")
    image_prompts: Dict[str, str] = Field(
        description="Dict of image prompts like {'image_1': 'prompt...', 'image_2': 'prompt...'}"
    )