from typing import List, Dict

def split_bytes(byte_data: bytes, max_size: int = 1024*10) -> List[bytes]:
    chunks = []  # Initialize an empty list to store the chunks
    for i in range(0, len(byte_data), max_size):  # Iterate over the byte data in chunks of max_size
        chunks.append(byte_data[i:i + max_size])  # Append each chunk to the list
    return chunks  # Return the list of chunks