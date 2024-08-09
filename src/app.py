#import google.generativeai as genai
import uvicorn
import dotenv
import os

from fastapi import FastAPI
from routes import model_comparision
from routes import benchmarks_tg_svc
from routes import benchmarks_qa_svc
from routes import validate
from routes import validate_m

# Load environment variables from .env file
dotenv.load_dotenv()

# Log in to Hugging Face Hub
from huggingface_hub import login
login(token=os.environ.get("HUGGINGFACE_TOKEN"))
#print(os.environ.get("HUGGINGFACE_TOKEN"))

# GOOGLE_GEMENI_API = os.environ.get('GOOGLE_GEMENI_API_KEY')
# genai.configure(api_key=GOOGLE_GEMENI_API)


app = FastAPI()

# # Include routers
app.include_router(validate.router)
app.include_router(validate_m.router)
app.include_router(model_comparision.router)
app.include_router(benchmarks_tg_svc.router)
app.include_router(benchmarks_qa_svc.router)


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)


