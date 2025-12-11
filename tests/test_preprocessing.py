import base64
from io import BytesIO

import pytest
from PIL import Image

from diagnose_w_confidence.server.preprocessing import (
    ContentImageURL,
    ContentText,
    ImageURL,
    Message,
    messages2prompt,
)


@pytest.fixture
def model_path() -> str:
    # warn: Qwen/Qwen2.5-VL-7B-Instruct has a invalid chat_template
    # which may omit `content: [{"text": "..."}, ...]` text fields.
    # you should use a local path instead.
    return "/slurm/models/Qwen2.5-VL-7B-Instruct"


@pytest.fixture
def b64_jpeg(width=256, height=256, color=(255, 0, 0)) -> str:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_text_only_messages_prompt(model_path: str):
    messages = [
        Message(role="user", content=[ContentText(type="text", text="Hello")]),
    ]
    out = messages2prompt(messages, model_path=model_path)
    assert "prompt" in out
    prompt: str = out["prompt"]
    assert prompt == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        "\n<|im_start|>user\nHello<|im_end|>"
        "\n<|im_start|>assistant\n"
    )


def test_multimodal_image_inserts_placeholder_and_returns_image(
    model_path: str, b64_jpeg: str
):
    messages = [
        Message(
            role="user",
            content=[
                ContentText(type="text", text="Describe the image."),
                ContentImageURL(type="image_url", image_url=b64_jpeg),
            ],
        ),
    ]
    messages = [Message.model_validate(m) for m in messages]

    # to debug what is inside messages
    # assert [m.model_dump() for m in messages] == []

    out = messages2prompt(messages, model_path=model_path)
    assert "prompt" in out and "multi_modal_data" in out

    prompt: str = out["prompt"]
    assert prompt == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        "\n<|im_start|>user\nDescribe the image.<|vision_start|><|image_pad|><|vision_end|><|im_end|>"
        "\n<|im_start|>assistant\n"
    )

    img = out["multi_modal_data"]["image"]
    assert isinstance(img, Image.Image)


def test_multimodal_with_data_uri_base64(model_path: str, b64_jpeg: str):
    data_uri = f"data:image/jpeg;base64,{b64_jpeg}"
    messages = [
        Message(
            role="user",
            content=[
                ContentImageURL(type="image_url", image_url=ImageURL(url=data_uri))
            ],
        )
    ]
    out = messages2prompt(messages, model_path=model_path)
    assert "prompt" in out and "multi_modal_data" in out

    prompt: str = out["prompt"]
    assert prompt == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>"
        "\n<|im_start|>assistant\n"
    )
    assert isinstance(out["multi_modal_data"]["image"], Image.Image)


def test_reject_if_http_url(model_path: str):
    messages = [
        Message(
            role="user",
            content=[
                ContentImageURL(
                    type="image_url",
                    image_url=ImageURL(url="https://example.com/a.png"),
                )
            ],
        )
    ]
    with pytest.raises(ValueError):
        messages2prompt(messages, model_path=model_path)


def test_reject_if_invalid_base64(model_path: str):
    messages = [
        Message(
            role="user",
            content=[ContentImageURL(type="image_url", image_url=ImageURL(url="@@@@"))],
        )
    ]
    with pytest.raises(ValueError):
        messages2prompt(messages, model_path=model_path)


if __name__ == "__main__":
    _model_path = model_path()
    b64_img = b64_jpeg()
    test_text_only_messages_prompt(_model_path)
    test_multimodal_image_inserts_placeholder_and_returns_image(_model_path, b64_img)
    test_multimodal_with_data_uri_base64(_model_path, b64_img)
