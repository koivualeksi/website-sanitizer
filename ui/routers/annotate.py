import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from markupsafe import escape
from pydantic import BaseModel, Field, model_validator

from db import annotations as db_ann
from scraper.converter import html_to_markdown, html_to_markdown_light
from ui.templating import templates

router = APIRouter()


class RangeItem(BaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def start_le_end(self):
        if self.start > self.end:
            raise ValueError("start must be <= end")
        return self


class AnnotationBody(BaseModel):
    ranges: list[RangeItem] = Field(default_factory=list, max_length=500)
    source: Literal["manual", "llm"] = "manual"


# /annotate/next MUST be registered before /annotate/{page_id} to avoid shadowing
@router.get("/annotate/next/{current_page_id}", response_class=HTMLResponse)
async def next_page(request: Request, current_page_id: int):
    pool = request.app.state.pool
    next_id = await db_ann.get_next_unvalidated(pool, current_page_id)
    if next_id is None:
        return HTMLResponse(
            '<div class="modal-done"><p>All pages have been reviewed!</p></div>',
            status_code=200,
        )
    return await get_modal(request, next_id)


@router.get("/annotate/{page_id}", response_class=HTMLResponse)
async def get_modal(request: Request, page_id: int):
    pool = request.app.state.pool
    data = await db_ann.get_annotation(pool, page_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Page not found")

    md = data["markdown"] or ""
    ranges = data["ranges"] or []
    source = data["source"] or "manual"

    return templates.TemplateResponse(
        request,
        "partials/modal.html",
        {
            "page_id": data["id"],
            "url": data["url"],
            "markdown": md,
            "ranges": ranges,
            "source": source,
            "validated": data["validated"] or False,
            "skipped": data["skipped"] or False,
            "has_html": bool(data["html"]),
            "has_cookies": data["has_cookies"],
            "tier": data["tier"] or "bronze",
        },
    )


@router.get("/annotate/{page_id}/structure", response_class=HTMLResponse)
async def get_structure(request: Request, page_id: int):
    pool = request.app.state.pool
    data = await db_ann.get_annotation(pool, page_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Page not found")

    raw_html = data["html"]
    if not raw_html:
        return HTMLResponse("<pre>No HTML available</pre>")

    structure = html_to_markdown(raw_html, base_url=data["url"])
    return HTMLResponse(f"<pre>{escape(structure)}</pre>")


@router.get("/annotate/{page_id}/structure-light", response_class=HTMLResponse)
async def get_structure_light(request: Request, page_id: int):
    pool = request.app.state.pool
    data = await db_ann.get_annotation(pool, page_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Page not found")

    raw_html = data["html"]
    if not raw_html:
        return HTMLResponse("<pre>No HTML available</pre>")

    structure = html_to_markdown_light(raw_html, base_url=data["url"])
    return HTMLResponse(f"<pre>{escape(structure)}</pre>")


_HTML_CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline' https: http:; "
    "img-src https: http: data:; "
    "font-src https: http:; "
    "script-src 'unsafe-inline'; "
    "connect-src 'none'"
)

_HTML_INJECT = (
    '<base target="_blank">'
    "<script>"
    'window.addEventListener("message",function(e){'
    'if(e.data==="getSelection"){'
    'e.source.postMessage({type:"selection",text:window.getSelection().toString()},"*");'
    "}"
    "});"
    "</script>"
)


@router.get("/annotate/{page_id}/html", response_class=HTMLResponse)
async def get_html(request: Request, page_id: int):
    pool = request.app.state.pool
    data = await db_ann.get_annotation(pool, page_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Page not found")

    raw_html = data["html"]
    if not raw_html:
        return HTMLResponse("<p>No HTML available</p>")

    if re.search(r"<head[\s>]", raw_html, re.IGNORECASE):
        html = re.sub(
            r"(<head[^>]*>)",
            lambda m: m.group(0) + _HTML_INJECT,
            raw_html, count=1, flags=re.IGNORECASE,
        )
    else:
        html = _HTML_INJECT + raw_html

    return HTMLResponse(html, headers={"Content-Security-Policy": _HTML_CSP})


@router.post("/annotate/{page_id}", response_class=HTMLResponse)
async def save(request: Request, page_id: int, body: AnnotationBody):
    pool = request.app.state.pool
    data = await db_ann.get_annotation(pool, page_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Page not found")

    ranges = [{"start": r.start, "end": r.end} for r in body.ranges]
    await db_ann.save_annotation(pool, page_id, ranges, "manual")
    return HTMLResponse("ok", status_code=200)


@router.post("/annotate/{page_id}/accept", response_class=HTMLResponse)
async def accept(request: Request, page_id: int):
    pool = request.app.state.pool
    data = await db_ann.get_annotation(pool, page_id)
    if data is None or data["ranges"] is None:
        raise HTTPException(status_code=404, detail="No annotation to accept")
    await db_ann.accept_annotation(pool, page_id)
    return HTMLResponse("ok", status_code=200)


@router.post("/annotate/{page_id}/skip", response_class=HTMLResponse)
async def skip(request: Request, page_id: int):
    pool = request.app.state.pool
    await db_ann.mark_skipped(pool, page_id)
    return HTMLResponse("ok", status_code=200)


@router.post("/annotate/{page_id}/cookie", response_class=HTMLResponse)
async def toggle_cookie(request: Request, page_id: int):
    pool = request.app.state.pool
    has_cookies = await db_ann.toggle_cookies(pool, page_id)
    return HTMLResponse("1" if has_cookies else "0", status_code=200)
