import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from psycopg import OperationalError

from db.pool import create_pool
from ui.routers import pages, annotate

UI_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(a: FastAPI):
    a.state.pool = create_pool()
    yield
    await asyncio.to_thread(a.state.pool.close)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")


@app.exception_handler(OperationalError)
async def db_error_handler(request: Request, exc: OperationalError):
    return HTMLResponse(
        content="<p>Database connection error. Check that PostgreSQL is running.</p>",
        status_code=503,
    )


app.include_router(pages.router)
app.include_router(annotate.router)
