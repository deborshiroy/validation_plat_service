from genericpath import exists
#from huggingface_hub import login
from utils.constants import *
import lm_eval
import subprocess
import torch 
import traceback
import os
import sys 
import gc

#import dotenv
#dotenv.load_dotenv("../.env")

class LMEvalRunner:
    def __init__(self):
        # print(os.environ.get("HUGGINGFACE_TOKEN"))
        # login(token=os.environ.get("HUGGINGFACE_TOKEN"))
        # self.path = '../output_files/temp_data_output'
        self.path = TMP_SAVE_UPLOAD_FILE_PATH

    def get_lm_eval_command(self, model_args):
        try:
            #Just like you can provide a local path to transformers.AutoModel, 
            #you can also provide a local path to lm_eval via --model_args pretrained=/path/to/model

            model_args="pretrained="+model_args+",trust_remote_code=True"
            
            print("Process started")
            if torch.cuda.is_available():
                self.device = "cuda:0" 
            else:
                self.device = "cpu"
            
            if os.path.exists(self.path):
                print("Output path exist")

                res= [
                "lm_eval",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", "hellaswag,squadv2",
                "--device", self.device,
                "--batch_size", "6", 
                "--limit", "3",
                "--output_path", self.path,
                "--log_samples"
                ] 
                

            else:
                print("Exception occurred while creating the folder")
                sys.exit(1)

            
            result= subprocess.run(res, check=True)
            print("Evaluation completed successfully.")
            return result
            
        
        except Exception as e:
            print(traceback.format_exc(),str(e))


    def clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()





    # def run_lm_eval(self,model_args):
    #     try:
    #         result= subprocess.run(self._get_lm_eval_command(model_args),check=True)
    #         print("lm_eval completed successfully.")
    #         print(result.stdout)
        
    #         #return result 

            
    #         #for path in os.scandir(self.path):
    #         #    if path.is_file():
    #         #        print(path.name)

    #     except subprocess.CalledProcessError as e:
    #        print(f"Error running lm_eval: {e}")

# Example usage
#model_args = "pretrained=microsoft/phi-2,trust_remote_code=True" or pretrained=openai-community/gpt2-large
# },trust_remote_code=True



