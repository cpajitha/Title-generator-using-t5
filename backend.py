from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import uvicorn

app = FastAPI()

# Load pretrained model and tokenizer from Hugging Face
model_name = "cpajitha/t5-small-finetuned-title_gen"
#model_name = "cpajitha/t5-small-finetuned-new-gettitle"
#tokenizer_name = "cpajitha/new_get_title"
tokenizer_name = "cpajitha/gettitle"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL
    allow_methods=["POST", "OPTIONS"],  # Allow POST and OPTIONS methods
    allow_headers=["*"],
)

@app.post("/generate-title")
async def generate_title(data: dict):
    abstract = data.get("abstract")
    if not abstract:
        return {"title": "Error: Missing abstract."}
    inputs = tokenizer(abstract, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, early_stopping=True)
    generated_title = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"title": generated_title}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
