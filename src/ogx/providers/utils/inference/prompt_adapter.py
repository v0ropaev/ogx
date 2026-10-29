# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import re
from typing import Any

import httpx2

from ogx.log import get_logger
from ogx.providers.utils.common.url_validation import validate_url_not_private
from ogx_api import (
    ImageContentItem,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIFile,
    TextContentItem,
)

log = get_logger(name=__name__, category="providers::utils")


def interleaved_content_as_str(
    content: Any,
    sep: str = " ",
) -> str:
    """Convert interleaved content items to a single string.

    Args:
        content: string, content item, or list of content items
        sep: separator between items when content is a list

    Returns:
        Concatenated string representation of the content
    """
    if content is None:
        return ""

    def _process(c) -> str:
        if isinstance(c, str):
            return c
        elif isinstance(c, TextContentItem) or isinstance(c, OpenAIChatCompletionContentPartTextParam):
            return c.text
        elif isinstance(c, ImageContentItem) or isinstance(c, OpenAIChatCompletionContentPartImageParam):
            return "<image>"
        elif isinstance(c, OpenAIFile):
            return "<file>"
        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if isinstance(content, list):
        return sep.join(_process(c) for c in content)
    else:
        return _process(content)


async def _validate_request_url_not_private(request: httpx2.Request) -> None:
    validate_url_not_private(str(request.url))


async def localize_image_content(uri: str) -> tuple[bytes, str] | None:
    """Download or decode image content from a URI.

    Args:
        uri: HTTP URL or data URI containing the image

    Returns:
        Tuple of (raw_bytes, format_string) or None if URI scheme is unsupported

    Raises:
        ValueError: If the URL, or any redirect target, resolves to a private address,
            or the download does not end in a 2xx response.
    """
    if uri.startswith("http"):
        # The request hook runs before the first request and before every redirect hop,
        # so a public URL cannot redirect the download to a private address.
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            event_hooks={"request": [_validate_request_url_not_private]},
        ) as client:
            r = await client.get(uri)
            if not r.is_success:
                raise ValueError(f"Failed to download image from {uri}: HTTP {r.status_code}")
            content = r.content
            content_type = r.headers.get("content-type")
            if content_type:
                format = content_type.split("/")[-1]
            else:
                format = "png"

        return content, format
    elif uri.startswith("data"):
        # data:image/{format};base64,{data}
        match = re.match(r"data:image/(\w+);base64,(.+)", uri)
        if not match:
            raise ValueError(f"Invalid data URL format, {uri[:40]}...")
        fmt, image_data = match.groups()
        content = base64.b64decode(image_data)
        return content, fmt
    else:
        return None
