from fastapi import FastAPI
import uvicorn
import sys 
import os 
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse 
from fastapi.responses import Response 
from src.textSummarizer.pipeline.predicition_pipeline import PredictionPipeline 

text:str = "Short stories date back to oral storytelling traditions which originally produced epics such as the Ramayana, the Mahabharata, and Homer's Iliad and Odyssey. Oral narratives were often told in the form of rhyming or rhythmic verse, often including recurring sections or, in the case of Homer, Homeric epithets. Such stylistic devices often acted as mnemonics for easier recall, rendition, and adaptation of the story. While the overall arc of the tale was told over the course of several performances, short sections of verse could focus on individual narratives that were the duration of a single telling. It may be helpful to classify such sections as oral short stories."

app= FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occured! {e}")

@app.post("/predict")
async def predict_route(text):
    try:
        obj= PredictionPipeline()
        text=obj.predict(text)
        return text
    except Exception as e:
        raise e

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 