import requests
from bs4 import BeautifulSoup
import gzip
import time
import os
import shutil

LOCAL = True
LOCAL_FILE = "C:/Users/leeka/Downloads/Kickstarter_2025-03-12T07_34_02_656Z.json.gz"
CHUNK_SIZE = 90 * 1024 * 1024  # 90MB chunks

def get_kickstarter_download_link():
    """
    Retrieves the download link from the Kickstarter datasets page
    """
    url = "https://webrobots.io/kickstarter-datasets/"
    
    try:
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the div with class fusion-text
        fusion_text_div = soup.find('div', class_='fusion-text')
        if not fusion_text_div:
            return None
        
        # Find the first ul in this div
        ul = fusion_text_div.find('ul')
        if not ul:
            return None
        
        # Find the first li in this ul
        li = ul.find('li')
        if not li:
            return None
        
        # Find all 'a' tags in the li and look for one with text containing "name json"
        for a in li.find_all('a'):
            if a.text and "json" in a.text.lower():
                print(f"Found link: {a['href']}")
                return a.get('href')
        
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def split_into_chunks(file_data, chunk_size):
    """
    Split the file data into chunks of specified size
    
    Args:
        file_data (bytes): The file data to chunk
        chunk_size (int): Size of each chunk in bytes
        
    Returns:
        list: List of file paths to the chunk files
    """
    chunk_files = []
    
    # Create a temporary directory for chunks if it doesn't exist
    temp_dir = "temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split the data into chunks and save each chunk
    for i in range(0, len(file_data), chunk_size):
        chunk = file_data[i:i + chunk_size]
        chunk_file_path = os.path.join(temp_dir, f"chunk_{i//chunk_size}.part")
        with open(chunk_file_path, "wb") as f:
            f.write(chunk)
        chunk_files.append(chunk_file_path)
        print(f"Created chunk {i//chunk_size} with size {len(chunk)/(1024*1024):.2f} MB")
    
    return chunk_files

def decompress_gzip_file(url, output_filename, keep_chunks=True):
    """
    Decompress a Gzip file using Python's standard gzip library.
    
    Args:
        url (str): URL or local path of the Gzip-compressed file.
        output_filename (str): Name of the file to save the decompressed data.
        keep_chunks (bool): Whether to keep the chunked files after decompression
    """

    if LOCAL:
        print("Using local file for testing.")
        with open(url, "rb") as f:
            compressed_bytes = f.read()
    else:
        print("Downloading the file...")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download the file: HTTP {response.status_code}")
            return
        compressed_bytes = response.content
        print(f"Downloaded {len(compressed_bytes)} bytes")
    
    print(f"Breaking file into chunks of {CHUNK_SIZE/(1024*1024):.2f} MB...")
    chunk_files = split_into_chunks(compressed_bytes, CHUNK_SIZE)
    
    print("Decompressing...")
    try:
        # Start the timer
        start_time = time.time()
        
        # Use the gzip module to decompress the entire file at once
        with gzip.open(url, "rb") as gz_file:
            with open(output_filename, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)
                
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Decompression completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Decompression failed: {e}")
        # Try alternative approach with the chunks
        print("Trying alternative decompression method with chunks...")
        try:
            start_time = time.time()
            
            # Create a temp file containing all chunks combined
            temp_combined = "temp_combined.gz"
            with open(temp_combined, "wb") as combined:
                for chunk_file in chunk_files:
                    with open(chunk_file, "rb") as f:
                        combined.write(f.read())
            
            # Decompress the combined file
            with gzip.open(temp_combined, "rb") as gz_file:
                with open(output_filename, "wb") as out_file:
                    shutil.copyfileobj(gz_file, out_file)
            
            # Remove temp combined file
            if os.path.exists(temp_combined):
                os.remove(temp_combined)
                
            elapsed_time = time.time() - start_time
            print(f"Alternative decompression completed in {elapsed_time:.2f} seconds")
            
        except Exception as inner_e:
            print(f"Alternative decompression also failed: {inner_e}")
            return
    
    # Clean up or move chunk files based on keep_chunks parameter
    if keep_chunks:
        # Create permanent directory for chunks if it doesn't exist
        chunks_dir = "gzip_chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Move chunk files to permanent directory
        for i, chunk_file in enumerate(chunk_files):
            if os.path.exists(chunk_file):
                dest_file = os.path.join(chunks_dir, f"chunk_{i}.part")
                shutil.move(chunk_file, dest_file)
        print(f"Chunk files have been saved to the '{chunks_dir}' directory")
    else:
        # Clean up temporary chunk files
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        
    # Remove temp directory if it's empty
    if os.path.exists("temp_chunks") and not os.listdir("temp_chunks"):
        shutil.rmtree("temp_chunks")

    print(f"Decompression successful. Data saved to {output_filename}")

if __name__ == "__main__":
    if LOCAL:
        url = LOCAL_FILE
    else:
        url = get_kickstarter_download_link()
    output_filename = "decompressed.json"
    decompress_gzip_file(url, output_filename, keep_chunks=True)