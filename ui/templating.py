from pathlib import Path

from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

UI_DIR = Path(__file__).parent

templates = Jinja2Templates(env=Environment(
    loader=FileSystemLoader(UI_DIR / "templates"),
    autoescape=True,
))
