
import time
from flask import Flask, request, jsonify
import os
import requests
import logging
from transformers import GPT2Tokenizer
from openai import OpenAI
from jsonschema import validate, ValidationError
import json
from dotenv import load_dotenv
load_dotenv()  # This will load environment variables from a .env file
import boto3
from botocore.exceptions import NoCredentialsError


# Setup logging based on the environment variable, defaulting to INFO
log_level_name = os.getenv('LOG_LEVEL', 'INFO')

# Create or get the logger
logger = logging.getLogger()

# Set the log level based on the environment
logger.setLevel(getattr(logging, log_level_name.upper(), logging.INFO))

# Create formatter including file name and line number
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(getattr(logging, log_level_name.upper(), logging.INFO))

ch.setFormatter(formatter)
logger.addHandler(ch)

# Initialize Flask app
app = Flask(__name__)

# Load GPT-2 Tokenizer from .env configuration
tokenizer_name = os.getenv('GPT2_TOKENIZER', 'gpt2')
try:
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise e

# Initialize OpenAI client with API key from environment
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise e

# Load schema for input validation from the current working directory
try:
    with open('request_schema.json', 'r') as schema_file:
        request_schema = json.load(schema_file)
except FileNotFoundError as e:
    logger.error(f"Schema file not found: {e}")
    raise e
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse schema file: {e}")
    raise e

# Load sample translation from environment variables
sample_translation_input = os.getenv('SAMPLE_TRANSLATION_INPUT', r'\section{Introducción a la Álgebra} \label{introAlg}')
sample_translation_output = os.getenv('SAMPLE_TRANSLATION_OUTPUT', r'\section{Introduction to Algebra} \label{introAlg}')

# Load pricing from environment variables
try:
    pricing_config = {
        "gpt-4o-mini": float(os.getenv("GPT_4O_MINI_PRICE", "0.00015")),
        "gpt-4o": float(os.getenv("GPT_4O_PRICE", "0.00250")),
        "o1-mini": float(os.getenv("O1_MINI_PRICE", "0.00300")),
        "gpt-3.5-turbo": float(os.getenv("GPT_3_5_TURBO_PRICE", "0.00300")),
        "gpt-4o-mini-output": float(os.getenv("GPT_4O_MINI_OUTPUT_PRICE", "0.000075")),
        "gpt-4o-output": float(os.getenv("GPT_4O_OUTPUT_PRICE", "0.00125")),
        "o1-mini-output": float(os.getenv("O1_MINI_OUTPUT_PRICE", "0.01200")),
        "gpt-3.5-turbo-output": float(os.getenv("GPT_3_5_TURBO_OUTPUT_PRICE", "0.00600"))
    }
except ValueError as e:
    logger.error(f"Failed to parse pricing information from environment variables: {e}")
    raise e

def split_large_chunks(chunks, ntokens, max_len=1000):
    """
    Split chunks that exceed the maximum token limit.

    Parameters:
    - chunks (list of str): List of original text chunks.
    - ntokens (list of int): List of token counts corresponding to each chunk.
    - max_len (int): Maximum allowable token length for any single chunk.

    Returns:
    - list of str: List of chunks where no chunk exceeds the maximum token length.
    """
    updated_chunks = []

    for chunk, ntoken in zip(chunks, ntokens):
        # If the chunk exceeds the max length, split it
        while ntoken > max_len:
            logger.warning(f"Splitting chunk due to excessive length ({ntoken} tokens > {max_len} token limit). Preview: '{chunk[:50]}...'")
            split_point = len(chunk) // 2  # Split the chunk into two halves

            # Split into two parts
            part1 = chunk[:split_point]
            part2 = chunk[split_point:]

            # Calculate tokens in each part
            part1_tokens = len(tokenizer.encode(part1))
            part2_tokens = len(tokenizer.encode(part2))

            # Add part1 to the updated chunks
            updated_chunks.append(part1)
            
            # Update chunk and ntoken for the next iteration
            chunk = part2
            ntoken = part2_tokens

        # Add the final part that is under the max_len limit
        updated_chunks.append(chunk)

    return updated_chunks

def create_overlapping_chunks(chunks, ntokens, max_len=1000, overlap_len=100):
    """
    Create chunks with overlapping tokens to ensure continuity in translation.

    Parameters:
    - chunks (list of str): List of original text chunks.
    - ntokens (list of int): List of token counts corresponding to each chunk.
    - max_len (int): Maximum allowable token length for any single chunk.
    - overlap_len (int): Number of tokens to overlap between consecutive chunks.

    Returns:
    - list of str: List of chunks with overlapping context.
    """
    overlapping_chunks = []
    cur_chunk = ""
    cur_tokens = 0

    for chunk, ntoken in zip(chunks, ntokens):
        if cur_tokens + ntoken <= max_len:
            cur_chunk += "\n\n" + chunk
            cur_tokens += ntoken
        else:
            # Add overlap from the current chunk to the previous chunk
            if overlapping_chunks:
                overlap_part = tokenizer.decode(tokenizer.encode(overlapping_chunks[-1])[-overlap_len:])
                cur_chunk = overlap_part + cur_chunk
            overlapping_chunks.append(cur_chunk)
            cur_chunk = chunk
            cur_tokens = ntoken

    if cur_chunk:
        overlapping_chunks.append(cur_chunk)

    return overlapping_chunks

def translate_chunk_with_token_count(chunk, model='gpt-4', dest_language='English', sample_translation=(
        sample_translation_input, sample_translation_output)):
    """
    Translate a chunk of LaTeX text while preserving LaTeX commands.

    Parameters:
    - chunk (str): The LaTeX text chunk to be translated.
    - model (str): The OpenAI model to use for translation.
    - dest_language (str): The target language for translation.
    - sample_translation (tuple): Sample LaTeX translation to provide context for the model.

    Returns:
    - str: The translated text.
    """
    prompt = f'''Translate the following LaTeX document into {dest_language}, translating only the plain text and leaving LaTeX commands unchanged.
    
Sample Translation Example:
Input:
{sample_translation[0]}
{chunk}
Output:
{sample_translation[1]}'''

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            top_p=1,
            max_tokens=1500,
        )
    except Exception as e:
        logger.error(f"Translation API request failed: {e}")
        raise e

    result = response.choices[0].message.content.strip()
    result = result.replace('"""', '')  # Remove the double quotes, as we used them to surround the text
    response_tokens = len(tokenizer.encode(result))
    return result, response_tokens
    

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate a LaTeX document from a given URL.

    Request Parameters (JSON):
    - file_url (str): URL of the LaTeX file to translate.
    - max_len (int, optional): Maximum token length for grouped chunks.
    - hard_max_len (int, optional): Hard limit for token length of individual chunks.
    - dest_language (str, optional): The target language for translation.
    - model (str, optional): The OpenAI model to use for translation.
    - num_pages (int, optional): The number of pages to translate.

    Returns:
    - JSON response indicating the success of the translation and the output file path.
    """
    # Extract parameters from the request
    request_data = request.get_json()
    try:
        validate(instance=request_data, schema=request_schema)
    except ValidationError as e:
        logger.error(f"Invalid request parameters: {e.message}")
        return jsonify({'error': f'Invalid request parameters: {e.message}'}), 400

    file_url = request_data.get('file_url')
    max_len = request_data.get('max_len', 1000)
    hard_max_len = request_data.get('hard_max_len', 1000)
    dest_language = request_data.get('dest_language')
    model = request_data.get('model', 'gpt-4')
    num_pages = request_data.get('num_pages', None)
    # Get cost per token from pricing config
    cost_per_input_token = pricing_config.get(model, 0.00002) / 1000000
    cost_per_output_token = pricing_config.get(f"{model}-output", 0.00002) / 1000000


    # Download LaTeX file from the provided URL
    try:
        response = requests.get(file_url)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f'Failed to download the file from the provided URL: {e}')
        return jsonify({'error': 'Failed to download the file from the provided URL'}), 400
    
    text = response.text

    # Split text into chunks
    chunks = text.split('\n\n')
    ntokens = [len(tokenizer.encode(chunk)) for chunk in chunks]
    
    # Step 1: Split large chunks to ensure no chunk exceeds the maximum token length
    chunks = split_large_chunks(chunks, ntokens, max_len=hard_max_len)

    # Step 2: Recalculate token counts after splitting
    ntokens = [len(tokenizer.encode(chunk)) for chunk in chunks]

    # Step 3: Create overlapping chunks for continuity
    overlapping_chunks = create_overlapping_chunks(chunks, ntokens, max_len=max_len)

    # Limit the number of pages to translate if specified
    if num_pages is not None:
        overlapping_chunks = overlapping_chunks[:num_pages]

    # Translate each chunk
    translated_chunks = []
    total_input_tokens = 0
    total_output_tokens = 0
    for i, chunk in enumerate(overlapping_chunks):
        logger.info(f"Translating chunk {i+1} / {len(overlapping_chunks)}")
        prompt_tokens = len(tokenizer.encode(chunk))  # Count input tokens for this chunk
        try:
            translated_text, response_tokens = translate_chunk_with_token_count(chunk, model=model, dest_language=dest_language)
            logger.info(f"translated_text {translated_text} ")
        except Exception as e:
            logger.error(f"Translation failed for chunk {i+1}: {e}")
            return jsonify({'error': f'Translation failed for chunk {i+1}: {e}'}), 500

        translated_chunks.append(translated_text)
        total_input_tokens += prompt_tokens
        total_output_tokens += response_tokens

    # Join translated chunks
    result = '\n\n'.join(translated_chunks)

    
     # Initialize S3 client
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    bucket_name = os.getenv('S3_BUCKET_NAME', 'micro3aiagents')
    object_key = f'translated_content_{int(time.time())}.tex'

    try:
        # Upload translated content to S3
        s3.put_object(Bucket=bucket_name, Key=object_key, Body=result, ContentType='text/plain')

        # Set the file to public
       # s3.put_object_acl(Bucket=bucket_name, Key=object_key, ACL='public-read')

        # Construct the file URL
        file_url = f'https://{bucket_name}.s3.us-east-1.amazonaws.com/{object_key}'
        
    except NoCredentialsError as e:
        logger.error(f'Credentials not available: {e}')
        return jsonify({'error': 'Failed to upload the translated file to S3 due to missing credentials'}), 500
    except Exception as e:
        logger.error(f'Failed to upload the translated file to S3: {e}')
        return jsonify({'error': 'Failed to upload the translated file to S3'}), 500
    
    # Calculate the total cost
    total_cost = (total_input_tokens * cost_per_input_token) + (total_output_tokens * cost_per_output_token)
    logger.info(f'Total input tokens used: {total_input_tokens}, Total output tokens used: {total_output_tokens}, Total cost: ${total_cost:.4f}')

    logger.info('Translation completed successfully.')
    return jsonify({
        'message': 'Translation completed successfully',
        'file_url': file_url,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_cost': f'${total_cost}'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
