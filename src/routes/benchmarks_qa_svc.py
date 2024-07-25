from fastapi import FastAPI, UploadFile ,File, HTTPException, status, APIRouter, UploadFile, Response, Form

from utils.data_models import *
from services.benchmarks_qa import LMEvalRunnerQA
from services.azure_services import *
from utils.constants import *

import shutil

router = APIRouter(prefix='/val/benchmark_qa',tags=['benchmark_qa'])

benchmark_eval_qa=LMEvalRunnerQA()



@router.post('/benchmark_validation_qa')
def benchmark_validation_qa(payload: model_validation_input_shcema_benchmark):
    '''
    Endpoint to receive an input model argument and generate metrics file.

    Args:
    - payload: The input file data for which metrics are to be generated.

    Returns:
    - FileResponse: A response containing the generated JSON file of metrics.
    '''
    try:
        
        payload = payload.dict()

        az = azure_ops(account_name = os.environ.get('ACCOUNT_NAME'),
                        account_key = os.environ.get('ACCOUNT_KEY'),
                        container_name = os.environ.get('CONTAINER_NAME'),
                        blob_path=os.environ.get('BENCHMARK_QNA')
                        )

        # Ensure the temporary save path exists
        os.makedirs(TMP_SAVE_UPLOAD_FILE_PATH, exist_ok=True)

        # Execute lm_eval command and get results
        results = benchmark_eval_qa.get_lm_eval_command_qa(payload['model_args'])

        # Extract the output file path
        for item in os.listdir(TMP_SAVE_UPLOAD_FILE_PATH):
            item_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH, item)
            if os.path.isdir(item_path):
                directory_path= item_path
        print(directory_path)


        for item in os.listdir(directory_path):
            if item.endswith('.json'):
                destination_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH,"openai-community__gpt2-large",item)
                
        #clean up the GPU memory
        benchmark_eval_qa.clean_memory()
       
        # Upload the result file to Azure Blob Storage
        sas_url = az.upload_file(local_file_path=destination_path, file_name="result.json")
        
        # Clean up temporary files
        # shutil.rmtree(TMP_SAVE_UPLOAD_FILE_PATH)
        
        # Close Azure connection
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