import random
# import tiktoken
import logging


def generate_random_hex(length=8):
    # Generate a random hexadecimal number of the specified length
    hex_number = ''.join(random.choices('0123456789ABCDEF', k=length))
    return hex_number
    

# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     print("Number_of_Tokens",num_tokens)
#     return num_tokens


# def file_to_str(file):
#     with open(file, "r") as file:
#         data = file.read()
#     return data


def Initialize_logger():
    '''
    # DEBUG: Detailed information, typically of interest only when diagnosing problems.

    # INFO: Confirmation that things are working as expected.

    # WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.

    # ERROR: Due to a more serious problem, the software has not been able to perform some function.

    # CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    '''
    logger= logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    #c_handler= logging.StreamHandler()
    #c_handler.setLevel(logging.INFO)

    f_handler= logging.FileHandler('logs\server.log')  #
    f_handler.setLevel(logging.INFO)

    c_format= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(c_format)

    logger.addHandler(f_handler)
    return logger 