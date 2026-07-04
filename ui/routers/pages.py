from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from db import annotations as db_ann
from ui.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    pool = request.app.state.pool
    tab = "unvalidated"
    rows, total = await db_ann.list_pages(pool, tab, 1, 25, "id", "asc", "")
    counts = await db_ann.get_tab_counts(pool)
    per_page = 25
    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(
        request,
        "list.html",
        {
            "rows": rows,
            "total": total,
            "page": 1,
            "per_page": per_page,
            "total_pages": total_pages,
            "tab": tab,
            "sort": "id",
            "dir": "asc",
            "search": "",
            "counts": counts,
        },
    )


@router.get("/pages", response_class=HTMLResponse)
async def pages_partial(
    request: Request,
    tab: Literal["unvalidated", "validated", "all"] = "unvalidated",
    page: int = Query(default=1, ge=1, le=10000),
    sort: Literal["id", "url", "source", "validated"] = "id",
    dir: Literal["asc", "desc"] = "asc",
    search: str = Query(default="", max_length=500),
):
    pool = request.app.state.pool
    per_page = 25
    rows, total = await db_ann.list_pages(pool, tab, page, per_page, sort, dir, search)
    counts = await db_ann.get_tab_counts(pool)
    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(
        request,
        "partials/table.html",
        {
            "rows": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "tab": tab,
            "sort": sort,
            "dir": dir,
            "search": search,
            "counts": counts,
        },
    )
