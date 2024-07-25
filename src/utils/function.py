import random
def generate_random_hex(length=8):
    # Generate a random hexadecimal number of the specified length
    hex_number = ''.join(random.choices('0123456789ABCDEF', k=length))
    return hex_number