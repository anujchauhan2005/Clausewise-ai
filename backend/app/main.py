"""
main.py

Entry point for the backend. Not much going on here on purpose -
just wiring up CORS (needed this once I started hitting the API from
the Streamlit frontend running on a different port, kept getting
blocked without it) and mounting the routes from api/routes.py.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # needs to happen before anything else imports os.getenv() for the model names / API key

# transformers auto-detects whatever ML backends are installed (torch, tf,
# flax) and tries to load all of them. we only use torch here, but if
# tensorflow happens to also be installed on the machine (common if it's
# a shared/older env) it drags in tensorflow + protobuf, which caused a
# nasty protobuf gencode/runtime version crash on my machine that had
# nothing to do with this project. forcing torch-only avoids that entirely.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="ClauseWise - Legal Document Intelligence API",
    description="Upload a contract and get entities, clause classification, risk flags, a summary, and Q&A.",
    version="1.0.0",
)

# wide open for now since this is just a portfolio project running
# locally / on a free host - would lock this down to specific origins
# if this were ever going to handle real documents from real users
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def health_check():
    # mostly just here so hitting the root URL doesn't 404, and so
    # Render/whatever host can ping this for uptime checks
    return {"status": "ok", "service": "clausewise-api"}
