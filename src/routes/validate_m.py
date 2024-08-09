from fastapi import FastAPI, UploadFile ,File, HTTPException, status, APIRouter, UploadFile, Response, Form
from fastapi.responses import FileResponse

from utils.data_models import *
from services.metrics_gemini import Evaluator_gemini
#from services.metrics_flant5 import Evaluator_flanT5
from services.azure_services import *
from utils.function import *
from utils.constants import *

import pandas as pd
import io
import os
import json
import shutil


router = APIRouter(prefix='/val_m/metrics_m',tags=['metrics_m']) 
logger = Initialize_logger()

evaluator_gemini = Evaluator_gemini()
#evaluator_flant5 = Evaluator_flanT5()


@router.post("/upload_csv_gemini")
async def upload_csv_gemini(file: UploadFile= File(...)):

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
            logger.error("No file uploaded")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
            


        # Ensure the temporary save path exists
        os.makedirs(TMP_SAVE_UPLOAD_FILE_PATH, exist_ok=True)
        logger.info("TMP_SAVE_UPLOAD_FILE_PATH created successfully")


        # Validate the file extension
        if file.filename.endswith('.csv'):
                print("Entered the file")
                # Read the file contents
                contents = await file.read()
                logger.info("Loading file data")

                # Process the contents (convert to DataFrame)
                df = pd.read_csv(io.BytesIO(contents))
                len_=len(df)
                list_of_metrics_result=[]
                list_of_metrics=[]
                for i in range(0,len_):
                    row=df.iloc[i].values
                    # question= row[0]
                    # answer= row[1]
                    # reference_answer= row[2]
                    # result_dict= evaluator_gpt.evaluate_all(question, answer, reference_answer)
                    
                    context= row[0]
                    question= row[1]
                    reference_answer= row[2]
                    logger.info(f"Row {i},inserted")

                    gen_answer_gemini = evaluator_gemini.generate_answer(context, question)
                    print(f"Answer: {gen_answer_gemini}")
                    result_gpt= evaluator_gemini.evaluate_all(gen_answer_gemini, reference_answer)
                    
                    list_of_metrics.append(result_gpt)
                    list_of_metrics_result.append({"result "+str(i+1):result_gpt})
                                      
                    
                metrics= evaluator_gemini.evaluate_average(list_of_metrics)
                list_of_metrics_result.append(metrics)
                logger.info("Metrics generated for Gemini model")

                # Define the path for the resulting JSON file
                json_file_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH, f"{file.filename.split('.')[0]}_metric_result.json")
                
                # Save the resulting metrics to the specified path
                with open(json_file_path, 'w') as json_file:
                    json.dump(list_of_metrics_result, json_file,indent=3 )
                    
                logger.info("JSON file created for Gemini model")
                print("json file created")

                sas_url = az.upload_file(local_file_path=json_file_path, file_name=f"{file.filename.split('.')[0]}_metrics.json")
                shutil.rmtree(TMP_SAVE_UPLOAD_FILE_PATH)
                logger.info("Result file uploaded to Azure")
                return {
                    "message": "successful",
                    "metric_file": sas_url
                }
                

            
        else:
            logger.error("Not provided a valid file")
            raise HTTPException(status_code=422, detail="Please provide a valid CSV file.")



    except HTTPException as http_exc:
        # Catch HTTP exceptions and re-raise them
        logger.error(f"exception:{http_exc}")
        raise http_exc

    except Exception as e:
        # Catch any other exceptions and return a 500 server error
        logger.error(f"Request not processed successfully. Details:{e}")
        raise HTTPException(status_code=500, detail=str(e))
        print(e)

    

'''

@router.post("/upload_csv_flant5")
async def upload_csv_flant5(file: UploadFile= File(...)):

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
            logger.error("No file uploaded")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

        # Ensure the temporary save path exists
        os.makedirs(TMP_SAVE_UPLOAD_FILE_PATH, exist_ok=True)
        logger.info("TMP_SAVE_UPLOAD_FILE_PATH created successfully")

        # Validate the file extension
        if file.filename.endswith('.csv'):
                print("Entered the file")
                # Read the file contents
                contents = await file.read()
                logger.info("Loading file data")

                # Process the contents (convert to DataFrame)
                df = pd.read_csv(io.BytesIO(contents))
                len_=len(df)
                list_of_metrics_result=[]
                list_of_metrics=[]
                for i in range(0,len_):
                    row=df.iloc[i].values
                    # question= row[0]
                    # answer= row[1]
                    # reference_answer= row[2]
                    # result_dict= evaluator_gpt.evaluate_all(question, answer, reference_answer)

                    context= row[0]
                    question= row[1]
                    reference_answer= row[2]
                    logger.info(f"Row {i},inserted")

                    gen_answer_flant5 = evaluator_flant5.generate_answer(context, question)
                    print(f"Answer: {gen_answer_flant5}")
                    result_flant5= evaluator_flant5.evaluate_all(gen_answer_flant5, reference_answer)
                    
                    list_of_metrics.append(result_flant5)
                    list_of_metrics_result.append({"result "+str(i+1):result_flant5})
                    #print(i+1,result_dict)
                    
                    
                metrics= evaluator_flant5.evaluate_average(list_of_metrics)
                list_of_metrics_result.append(metrics)
                logger.info("Metrics generated for FlanT5 model")


                # Define the path for the resulting JSON file
                json_file_path = os.path.join(TMP_SAVE_UPLOAD_FILE_PATH, f"{file.filename.split('.')[0]}_metric_result.json")
                
                # Save the resulting metrics to the specified path
                with open(json_file_path, 'w') as json_file:
                    json.dump(list_of_metrics_result, json_file,indent=3 )
                    
                logger.info("JSON file created for flanT5 model")
                print("json file created")

                sas_url = az.upload_file(local_file_path=json_file_path, file_name=f"{file.filename.split('.')[0]}_metrics.json")
                shutil.rmtree(TMP_SAVE_UPLOAD_FILE_PATH)
                logger.info("Result file uploaded to Azure")
                return {
                    "message": "successful",
                    "metric_file": sas_url
                }
                

            
        else:
            logger.error("Not provided a valid file")
            raise HTTPException(status_code=422, detail="Please provide a valid CSV file.")



    except HTTPException as http_exc:
        # Catch HTTP exceptions and re-raise them
        logger.error(f"exception:{http_exc}")
        raise http_exc

    except Exception as e:
        # Catch any other exceptions and return a 500 server error
        logger.error(f"Request not processed successfully. Details:{e}")
        raise HTTPException(status_code=500, detail=str(e))
        print(e)
'''




@router.post('/model_validation_gemini')
def model_validation_gemini(payload: model_validation_input_shcema_text):
    try:
        
        payload = payload.dict()
        logger.info(f"Payload: {payload}")
        if not payload:
            logger.error("Payload not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="payload not found")

        gen_answer_gemini = evaluator_gemini.generate_answer(payload['context'], payload['question'])
        #print(f"Answer: {gen_answer_gpt}")

        if not gen_answer_gemini:
            logger.error("Generated response not found")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generated response not found")
        logger.info(f"Answer: {gen_answer_gemini}")

        metrics_gemini = evaluator_gemini.evaluate_all( gen_answer_gemini, payload['reference_answer'])
        logger.info("Metrics computed successfully for Gemini")
        return {
            "message": "successful",
            "metrics": metrics_gemini
        }
    except Exception as e:
        logger.error(f"Exception in generated response:{e}")
        print(traceback.format_exc())



'''
@router.post('/model_validation_flant5')
def model_validation_flant5(payload: model_validation_input_shcema_text):
    try:
        payload = payload.dict()
        print("Entered API")
        print("data are as follows:")
        print("context",payload['context'])
        print("question",payload['question'])
        print("reference_answer",payload['reference_answer'])

        logger.info(f"Payload: {payload}")
        if not payload:
            logger.error("Payload not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="payload not found")

        gen_answer_flant5 = evaluator_flant5.generate_answer(payload['context'], payload['question'])
        if not gen_answer_flant5:
            logger.error("Generated response not found")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generated response not found")
        logger.info(f"Answer: {gen_answer_flant5}")
        print(f"Answer: {gen_answer_flant5}")

        metrics_flant5 = evaluator_flant5.evaluate_all( gen_answer_flant5, payload['reference_answer'])
        logger.info("Metrics computed successfully for FlanT5")
        print("Flant5 metrics computed")
        return {
            "message": "successful",
            "metrics": metrics_flant5
        }
    except Exception as e:
        logger.error(f"Exception in generated response:{e}")
        print("traceback.format_exc()")
       
'''      
    

        

    
        


