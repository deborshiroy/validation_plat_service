import pandas as pd
import json
import io

from fastapi import FastAPI, UploadFile ,File, HTTPException, status, APIRouter, UploadFile, Response, Form
from fastapi.responses import FileResponse
from pandas import json_normalize
from IPython.display import display

from utils.function import *
from utils.data_models import *
from services.azure_services import *
from utils.constants import *
from services.metrics_gpt import Evaluator_gpt
from services.metrics_flant5 import Evaluator_flanT5
from services.metrics_gemini import Evaluator_gemini

router = APIRouter(prefix='/val/combined_metrics',tags=['combined_metrics']) 
logger= Initialize_logger()

evaluator_gpt= Evaluator_gpt()
evaluator_flant5= Evaluator_flanT5()
evaluator_gemini= Evaluator_gemini()


@router.post('/model_comparison_text')
def model_comparison_text(payload: model_validation_input_shcema_text):
    try:
        payload = payload.dict()
        logger.info(f"Payload: {payload}")
        if not payload:
            logger.error("Payload not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="payload not found")

        print("Input loaded")



        gen_answer_gpt = evaluator_gpt.generate_answer(payload['context'], payload['question'])
        if not gen_answer_gpt:
            logger.error("Generated response not found")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generated response not found")
        logger.info(f"Answer: {gen_answer_gpt}")
        
        print(f"Answer: {gen_answer_gpt}")

        metrics_gpt = evaluator_gpt.evaluate_all( gen_answer_gpt, payload['reference_answer'])
        metrics_gpt.update({"Base Cost": "0.26 Rs/ 1M Tokens"})
        logger.info("Metrics computed successfully for GPT")
        
        print("GPT metrics computed")



        gen_answer_flant5 = evaluator_flant5.generate_answer(payload['context'], payload['question'])
        if not gen_answer_flant5:
            logger.error("Generated response not found")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generated response not found")
        logger.info(f"Answer: {gen_answer_flant5}")
        
        print(f"Answer: {gen_answer_flant5}")
        
        metrics_flant5 = evaluator_flant5.evaluate_all(gen_answer_flant5, payload['reference_answer'])
        metrics_flant5.update({"Base Cost": "0 Rs/ 1M Tokens"})
        logger.info("Metrics computed successfully for FlanT5")
        print("flant5 metrics computed")



        gen_answer_gemini = evaluator_gemini.generate_answer(payload['context'], payload['question'])
        if not gen_answer_gemini:
            logger.error("Generated response not found")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generated response not found")
        logger.info(f"Answer: {gen_answer_gemini}")
        
        print(f"Answer: {gen_answer_gemini}")
        
        metrics_gemini = evaluator_gemini.evaluate_all(gen_answer_gemini, payload['reference_answer'])
        metrics_gemini.update({"Base Cost": "0 Rs/ 1M Tokens"})
        logger.info("Metrics computed successfully for Gemini")
        print("Gemini metrics computed")



        json_data= {"FlanT5 Base": metrics_flant5, "GPT2": metrics_gpt, "Gemini": metrics_gemini }
        logger.info("Evaluation Completed")
        
        print ("evaluation completed")

        # Use pandas.DataFrame.from_dict() to Convert JSON to DataFrame
        df = pd.DataFrame(json_data)
        result= df.transpose()

        # # Convert data to dict
        # data = json.loads(payload)
        
        # # Convert dict to string
        # data = json.dumps(payload)
        # number_of_tokens= num_tokens_from_string(payload,"cl100k_base")
        # print(number_of_tokens)

        print(result)
        #print(display(result))
        
        return df
        
    except Exception as e:
        logger.error(f"Exception in generated response:{e}")
        print("traceback.format_exc()")




@router.post("/model_comparison_csv")
async def model_comparison_csv(file: UploadFile= File(...)):  
    try:

        if not file:
            logger.error("No file uploaded")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

        # Validate the file extension
        if file.filename.endswith('.csv'):
            print("In the file")
            # Read the file contents
            contents = await file.read()
            logger.info("Loading file data")

            # Process the contents (convert to DataFrame)
            df = pd.read_csv(io.BytesIO(contents))
            len_=len(df)

            # data= file_to_str(file.filename)
            # number_of_tokens= num_tokens_from_string(data,"cl100k_base")

            print("reading the file")
            list_of_metrics_gpt=[]
            list_of_metrics_flant5=[]
            list_of_metrics_gemini=[]

            for i in range(0,len_):
                row=df.iloc[i].values
                context= row[0]
                question= row[1]
                reference_answer= row[2]
                logger.info(f"Row {i},inserted")

                gen_answer_gpt = evaluator_gpt.generate_answer(context, question)
                print(f"Answer: {gen_answer_gpt}")
                result_gpt= evaluator_gpt.evaluate_all(gen_answer_gpt, reference_answer)
                list_of_metrics_gpt.append(result_gpt)


                gen_answer_flant5 = evaluator_flant5.generate_answer(context, question)
                print(f"Answer: {gen_answer_flant5}")
                result_flant5= evaluator_flant5.evaluate_all(gen_answer_flant5, reference_answer)
                list_of_metrics_flant5.append(result_flant5)


                gen_answer_gemini = evaluator_gemini.generate_answer(context, question)
                print(f"Answer: {gen_answer_gemini}")
                result_gemini= evaluator_gemini.evaluate_all(gen_answer_gemini, reference_answer)
                list_of_metrics_gemini.append(result_gemini)


            print("Evaluation completed")
            metrics_gpt= evaluator_gpt.evaluate_average(list_of_metrics_gpt)
            metrics_gpt["Average result"].update({"Base Cost": "0.26 Rs/ 1M Tokens"})

            metrics_flant5= evaluator_flant5.evaluate_average(list_of_metrics_flant5)
            metrics_flant5["Average result"].update({"Base Cost": "0 Rs/ 1M Tokens"})

            metrics_gemini= evaluator_flant5.evaluate_average(list_of_metrics_gemini)
            metrics_gemini["Average result"].update({"Base Cost": "0 Rs/ 1M Tokens"})

            logger.info("Metrics generated")


            json_data= {"FlanT5 Base": metrics_flant5["Average result"], "GPT2": metrics_gpt["Average result"], "Gemini": metrics_gemini["Average result"]}
            logger.info(f"Metrics:{json_data}")

            print(json_data)

            # Use pandas.DataFrame.from_dict() to Convert JSON to DataFrame
            df = pd.DataFrame(json_data)
            result= df.transpose()
            print(result)
            return df

            # for i in range(0,len_):
            #     row=df.iloc[i].values
            #     question= row[0]
            #     answer= row[1]
            #     reference_answer= row[2]
            #     result_flant5= evaluator_flant5.evaluate_all(question, answer, reference_answer)
            #     list_of_metrics.append(result_flant5)
                
            

            
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



