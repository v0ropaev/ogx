# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable

import httpx2
import pytest

from ogx.providers.utils.inference.prompt_adapter import localize_image_content

# IP literal hosts keep validate_url_not_private off DNS, so the real SSRF check runs.
PUBLIC_HOST = "http://93.184.216.34"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"

Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
def serve(monkeypatch) -> Callable[[Handler], list[httpx2.Request]]:
    """Return an installer that routes localize_image_content's httpx2 client through a MockTransport.

    The installer takes a request handler and returns the list of requests the client sends.
    """
    real_client = httpx2.AsyncClient
    captured: list[httpx2.Request] = []

    def install(handler: Handler) -> list[httpx2.Request]:
        def recording_handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return handler(request)

        def make_client(*args, **kwargs):
            return real_client(*args, transport=httpx2.MockTransport(recording_handler), **kwargs)

        monkeypatch.setattr(httpx2, "AsyncClient", make_client)
        return captured

    return install


async def test_downloads_image(serve):
    serve(lambda request: httpx2.Response(200, headers={"content-type": "image/png"}, content=PNG_BYTES))

    assert await localize_image_content(f"{PUBLIC_HOST}/cat.png") == (PNG_BYTES, "png")


@pytest.mark.parametrize("status_code", [403, 404, 429, 500])
async def test_rejects_error_response(serve, status_code):
    serve(lambda request: httpx2.Response(status_code, headers={"content-type": "text/html"}, content=b"<html></html>"))

    with pytest.raises(ValueError, match=f"Failed to download image from .*: HTTP {status_code}"):
        await localize_image_content(f"{PUBLIC_HOST}/missing.png")


async def test_follows_redirect(serve):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/moved.png":
            return httpx2.Response(302, headers={"location": f"{PUBLIC_HOST}/cat.png"})
        return httpx2.Response(200, headers={"content-type": "image/png"}, content=PNG_BYTES)

    captured = serve(handler)

    assert await localize_image_content(f"{PUBLIC_HOST}/moved.png") == (PNG_BYTES, "png")
    assert [request.url.path for request in captured] == ["/moved.png", "/cat.png"]


async def test_rejects_redirect_to_private_address(serve):
    captured = serve(
        lambda request: httpx2.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
    )

    with pytest.raises(ValueError, match="private IP"):
        await localize_image_content(f"{PUBLIC_HOST}/moved.png")
    assert [request.url.host for request in captured] == ["93.184.216.34"]


async def test_rejects_private_url_without_request(serve):
    captured = serve(lambda request: httpx2.Response(200, headers={"content-type": "image/png"}, content=PNG_BYTES))

    with pytest.raises(ValueError, match="private IP"):
        await localize_image_content("http://127.0.0.1/cat.png")
    assert captured == []
