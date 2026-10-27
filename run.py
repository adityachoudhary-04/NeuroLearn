import uvicorn
import webbrowser
import threading
import time
import os

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000/")

if __name__ == "__main__":
    print("=" * 65)
    print(" NeuroLearn: PDF RAG & Q/A Learning Assistant")
    print(" Server starting at: http://127.0.0.1:8000")
    print(" API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 65)
    
    # Optionally open browser automatically
    threading.Thread(target=open_browser, daemon=True).start()
    
    # A reload worker can fail to spawn on some Windows/VS Code setups and
    # leave a cached-looking UI with no API behind it. Keep the default launch
    # single-process and predictable. Developers can opt in with
    # NEUROLEARN_RELOAD=1 when they specifically need hot reload.
    enable_reload = os.getenv("NEUROLEARN_RELOAD", "0") == "1"
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=enable_reload)
