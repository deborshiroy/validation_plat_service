from fastapi import FastAPI, HTTPException, status, APIRouter, UploadFile, File, Response, Form

from utils.data_models import *
from services.text_generation.metrics import Evaluator


router = APIRouter()

evaluator = Evaluator()

@router.post('/model_validation')
def model_validation(payload: model_validation_input_shcema)
    payload = payload.dict()
    metrics = evaluator.evaluate_all(question, response, reference)
    return {
        "message": "successful",
        "metrics": metrics
    }
    
    

    
        


