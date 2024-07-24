from fastapi import FastAPI
from routes import validate
from routes import benchmarks_tg_svc
from routes import benchmarks_qa_svc
import uvicorn

import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Log in to Hugging Face Hub
from huggingface_hub import login
print(os.environ.get("HUGGINGFACE_TOKEN"))
login(token=os.environ.get("HUGGINGFACE_TOKEN"))

app = FastAPI()

# Include routers
app.include_router(validate.router)
app.include_router(benchmarks_tg_svc.router)
app.include_router(benchmarks_qa_svc.router)


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)






# from services.benchmarks_tg import *

# if __name__ == "__main__":
#     test=LMEvalRunner()
#     model_args= input("Enter the model name or path: ")
#     test.get_lm_eval_command(model_args)
#     test.clean_memory()




