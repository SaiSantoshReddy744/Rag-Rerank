"""Single startup command:  python run.py   ->  http://127.0.0.1:8000"""
import threading
import webbrowser

import uvicorn

if __name__ == "__main__":
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
