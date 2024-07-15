from pydantic import BaseModel

class model_validation_input_shcema(BaseModel):
    question: str
    answer: str
    reference_answer: str