from fastapi import FastAPI, UploadFile ,File, HTTPException, status, APIRouter, UploadFile, Response, Form
from fastapi.responses import FileResponse
 
from utils.data_models import *
from services.metrics import Evaluator
from services.azure_services import *
from utils.constants import *
 
import pandas as pd
import io
import os
import json
import shutil
 
 
router = APIRouter(prefix='/val/metrics',tags=['metrics'])
 
evaluator = Evaluator()
 
@router.post("/upload_csv")
async def upload_csv(file: UploadFile= File(...)):
 
    """
    Endpoint to receive an input file and generate metrics.
    It accepts a save path and a csv file, generates metrics, and returns the resulting JSON file.
 
    Args:
    - file (UploadFile): The input file data for which metrics are to be generated.
 
    Returns:
    - FileResponse: A response containing the generated JSON file of metrics.
    """
 
    try:
 
        az = azure_ops(account_name = os.environ.get('ACCOUNT_NAME'),
                        account_key = os.environ.get('ACCOUNT_KEY'),
                        container_name = os.environ.get('CONTAINER_NAME'),
                        blob_path=os.environ.get('VALIDATION')
                        )
       
        if not file:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
 
        # Ensure the temporary save path exists
        os.makedirs(TMP_SAVE_UPLOAD_FILE_PATH, exist_ok=True)
 
        # Validate the file extension
        if file.filename.endswith('.csv'):
                print("Entered the file")
                # Read the file contents
                contents = await file.read()
 
                # Process the contents (convert to DataFrame)
                df = pd.read_csv(io.BytesIO(contents))
                len_=len(df)
                list_of_metrics_result=[]
                list_of_metrics=[]
                for i in range(0,len_):
                    row=df.iloc[i].values
                    question= row[0]
                    answer= row[1]
                    reference_answer= row[2]
                    result_dict= evaluator.evaluate_all(question, answer, reference_answer)
                    list_of_metrics.append(result_dict)
                    list_of_metrics_result.append({"result "+str(i+1):result_dict})
                    print(i+1,result_dict)
                   
                   
                metrics= evaluator.evaluate_average(list_of_metrics)
                list_of_metrics_result.append(metrics)
 
                print(metrics)
 
                # Define the path for the resulting JSON file
                json_file_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH, f"{file.filename.split('.')[0]}_metric_result.json")
               
                # Save the embeddings to the specified path
                with open(json_file_path, 'w') as json_file:
                    json.dump(list_of_metrics_result, json_file,indent=3 )
                   
                print("json file created")
 
                sas_url = az.upload_file(local_file_path=json_file_path, file_name=f"{file.filename.split('.')[0]}_metrics.json")
                shutil.rmtree(TMP_SAVE_UPLOAD_FILE_PATH)
                return {
                    "message": "successful",
                    "metric_file": sas_url
                }
 
           
        else:
            raise HTTPException(status_code=422, detail="Please provide a valid CSV file.")
 
 
 
    except HTTPException as http_exc:
        # Catch HTTP exceptions and re-raise them
        raise http_exc
    except Exception as e:
        # Catch any other exceptions and return a 500 server error
        raise HTTPException(status_code=500, detail=str(e))
        print(e)
   
 
 
 
@router.post('/model_validation')
def model_validation(payload: model_validation_input_shcema_text):
    try:
       
        payload = payload.dict()
        metrics = evaluator.evaluate_all(payload['question'], payload['answer'], payload['reference_answer'])
        return {
            "message": "successful",
            "metrics": metrics
        }
    except Exception as e:
        print("traceback.format_exc()")