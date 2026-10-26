"""
Hugging Face Spaces Entrypoint (Gradio SDK compatible)
Runs the Enterprise AURA CRAG FastAPI application with custom Glass-Box Frontend
and mounts Gradio on port 7860 with 16 GB RAM support.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-download NLTK sentence tokenizer models if missing
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

import gradio as gr
import uvicorn
from api.main import app

# Minimal Gradio helper block mounted at /gradio
with gr.Blocks(title="AURA CRAG Studio") as demo:
    gr.Markdown("# ⚡ AURA CRAG — Enterprise Financial Intelligence")
    gr.Markdown(
        "Interactive Glass-Box Studio is active! "
        "[**Click here to launch the Full Glass-Box Studio**](/) "
        "or access the [**Swagger OpenAPI Docs**](/docs)."
    )

# Mount Gradio onto the existing FastAPI application
# Root '/' continues to serve our custom warm Split-Screen Studio UI!
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
