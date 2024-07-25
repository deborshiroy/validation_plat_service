from fastapi import FastAPI, UploadFile ,File, HTTPException, status, APIRouter, UploadFile, Response, Form

from utils.data_models import *
from services.benchmarks_tg import LMEvalRunner
from services.azure_services import *
from utils.constants import *
from utils.function import * 

import shutil

router = APIRouter(prefix='/val/benchmark_tg',tags=['benchmark_tg'])

benchmark_eval=LMEvalRunner()



@router.post('/benchmark_validation')
def benchmark_validation(payload: model_validation_input_shcema_benchmark):
    try:
        rand_num = generate_random_hex()
        payload = payload.dict()

        az = azure_ops(account_name = os.environ.get('ACCOUNT_NAME'),
                        account_key = os.environ.get('ACCOUNT_KEY'),
                        container_name = os.environ.get('CONTAINER_NAME'),
                        blob_path=os.environ.get('BENCHMARK_TG')
                        )

        # Ensure the temporary save path exists
        os.makedirs(TMP_SAVE_UPLOAD_FILE_PATH, exist_ok=True)


        results = benchmark_eval.get_lm_eval_command(payload['model_args'])   
        file_name = benchmark_eval.file_name 
        print(file_name)   
        benchmark_eval.clean_memory()
        output_file_path = file_name+'/'+payload['model_args'].replace('/', '__')+'/'+[i for i in os.listdir(os.path.join(file_name, payload['model_args'].replace('/', '__'))) if i.startswith('results') and i.endswith('.json')][0]
        print(output_file_path)
        sas_url = az.upload_file(local_file_path=output_file_path, file_name="result.json")
        
        az.azure_close_conn()
        shutil.rmtree(file_name)
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