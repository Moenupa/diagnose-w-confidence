import base64
from io import BytesIO
from typing import Literal, Union

from PIL import Image
from pydantic import BaseModel


class ImageURL(BaseModel):
    url: str  # accept base64 data URIs or raw base64 strings


class ContentText(BaseModel):
    type: Literal["text"]
    text: str


class ContentImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: Union[ImageURL, str]


class Message(BaseModel):
    role: str
    content: str | list[ContentText | ContentImageURL]


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


# TODO: modify this function to use model's chat template
def messages2prompt(messages: list[Message], model_path: str) -> dict:
    from transformers import AutoTokenizer  # defer import to avoid global dependency

    # Prepare chat messages in HF format and collect images
    images_for_prompt: list[Image.Image] = []

    for m in messages:
        content_val = m.content

        if not isinstance(content_val, (str, list)):
            raise RuntimeError(
                f"Invalid internal pydantic parsing: type(content)={type(content_val)} not in (str, list)"
            )

        # Normalize to list of segments with optional image placeholders
        segments: list[str] = []
        if isinstance(content_val, str):
            continue
        for part in content_val:
            if not isinstance(part, (dict, ContentText, ContentImageURL)):
                raise RuntimeError(
                    f"Invalid internal pydantic parsing: type(part)={type(part)} not in (dict, ContentText, ContentImageURL)"
                )

            if part.type == "text":
                continue

            assert part.type == "image_url", (
                "Only image_url content type is supported for multimodal inputs."
            )
            image_url = part.image_url
            url = image_url if isinstance(image_url, str) else image_url.url
            if url.startswith("http://") or url.startswith("https://"):
                # not implemented
                raise ValueError(
                    "Remote image URLs are not supported. Use base64 data URIs."
                )

            b64_payload = url
            if url.startswith("data:"):
                i = url.find("base64,")
                if i != -1:
                    b64_payload = url[i + 7 :]
            try:
                raw = base64.b64decode(b64_payload)
                pil = Image.open(BytesIO(raw)).convert("RGB")
                images_for_prompt.append(pil)
                # Insert an image placeholder token expected by most chat templates
                segments.append("<image>")
            except Exception as e:
                raise ValueError("Invalid base64 image data.") from e

    # Apply the model's chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    out = {"prompt": prompt}
    if images_for_prompt:
        out |= {
            "multi_modal_data": {
                "image": images_for_prompt[0]
                if len(images_for_prompt) == 1
                else images_for_prompt
            }
        }

    if not set(out.keys()) <= {"prompt", "multi_modal_data"}:
        raise RuntimeError(f"Invalid prompt dict keys: {out.keys()}")
    return out
