from fastapi import FastAPI, UploadFile ,File, HTTPException, status, APIRouter, UploadFile, Response, Form

from utils.data_models import *
from services.benchmarks_tg import LMEvalRunner
from services.azure_services import *
from utils.constants import *

import shutil

router = APIRouter(prefix='/val/benchmark_tg',tags=['benchmark_tg'])

benchmark_eval=LMEvalRunner()



@router.post('/benchmark_validation')
def benchmark_validation(payload: model_validation_input_shcema_benchmark):
    try:
        
        payload = payload.dict()

        az = azure_ops(account_name = os.environ.get('ACCOUNT_NAME'),
                        account_key = os.environ.get('ACCOUNT_KEY'),
                        container_name = os.environ.get('CONTAINER_NAME'),
                        blob_path=os.environ.get('BENCHMARK_TG')
                        )

        # Ensure the temporary save path exists
        os.makedirs(TMP_SAVE_UPLOAD_FILE_PATH, exist_ok=True)


        results = benchmark_eval.get_lm_eval_command(payload['model_args'])
        

        for item in os.listdir(TMP_SAVE_UPLOAD_FILE_PATH):
            item_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH, item)
            if os.path.isdir(item_path):
                directory_path= item_path


        for item in os.listdir(directory_path):
            if item.endswith('.json'):
                destination_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH,"openai-community__gpt2-large",item)
                
        
        benchmark_eval.clean_memory()
       
        
        sas_url = az.upload_file(local_file_path=destination_path, file_name="result.json")
        print()
        shutil.rmtree(TMP_SAVE_UPLOAD_FILE_PATH)
        az.azure_close_conn()
        return {
            "message": "successful",
            "output_file": sas_url
        }
        
    except HTTPException as http_exc:
        # Catch HTTP exceptions and re-raise them
        raise http_exc
    except Exception as e:
        # Catch any other exceptions and return a 500 server error
        raise HTTPException(status_code=500, detail=str(e))
        print(e)