from fastapi import FastAPI, HTTPException, status, APIRouter, UploadFile, File, Response, Form

from utils.data_models import *
from services.metrics import Evaluator


router = APIRouter()

evaluator = Evaluator()

@router.post('/model_validation')
def model_validation(payload: model_validation_input_shcema):
    try:
        
        payload = payload.dict()
        metrics = evaluator.evaluate_all(payload['question'], payload['answer'], payload['reference_answer'])
        return {
            "message": "successful",
            "metrics": metrics
        }
    except Exception as e:
        print(traceback.format_exc())
        
    

    
        


