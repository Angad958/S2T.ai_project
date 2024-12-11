from fastapi import FastAPI, Response, HTTPException,APIRouter,Request
from services.sentiment import ask_question
from services.rag_pipeline import search_DB
import requests
app = FastAPI()

@app.get("/")
def home():
    return {"hello": "world"}


@app.post("/ask")
async def search(request: Request):
    body = await request.json()
    query = body.get("query")
    
    if not query:
        raise HTTPException(status_code=400, detail="Missing query parameter")

    # Call the search_DB function and return the results
    data = await ask_question(query)
    return {"response": data}


@app.post("/getCricketInfo")
async def searchCrickter(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query parameter")
    # Call the search_DB function and return the results
    data = search_DB(query)
    return {"response": data}
