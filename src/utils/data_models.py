from pydantic import BaseModel

class model_validation_input_shcema(BaseModel):
    question: str
    answer: str
    reference_answer: str
    
from pydantic import BaseModel

class model_validation_input_shcema_text(BaseModel):
    """
    Represents input data for text-based model validation.
    """
    question: str
    answer: str
    reference_answer: str


class model_validation_input_shcema_benchmark(BaseModel):
    """
    Represents input data for benchmarking a model.
    """
    model_args: str
    
class metrics_az_schema(BaseModel):
    """
    Represents Azure metrics configuration.
    """
    account_name: str
    account_key: str
    container_name: str
    blob_path: str
    file_name: str