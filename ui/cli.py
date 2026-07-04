import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv()
    uvicorn.run("ui.app:app", host="127.0.0.1", port=8000, reload=True)
