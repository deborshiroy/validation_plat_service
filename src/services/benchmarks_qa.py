from genericpath import exists
from utils.constants import *
import lm_eval
import subprocess
import torch 
import traceback
import os
import sys 
import gc


class LMEvalRunnerQA:
    def __init__(self):
        """
        Initializes an instance of LMEvalRunner.

        Attributes:
            path (str): Path for temporary file storage.
        """
        self.path = TMP_SAVE_UPLOAD_FILE_PATH
       
        

    def get_lm_eval_command_qa(self, model_args):
        """
        Executes the lm_eval command for evaluating a pre-trained language model.

        Args:
            model_args (str): Arguments for the pre-trained model.

        Returns:
            CompletedProcess: Result of the evaluation process.
        """
        try:
            
            model_args="pretrained="+model_args +",trust_remote_code=True"
            
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
                "--tasks", "squad_completion",     #hellaswag, basque-glue
                "--device", self.device,
                "--batch_size", "8", 
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
        """
        Clears GPU memory and performs garbage collection.
        """
        torch.cuda.empty_cache()
        gc.collect()





