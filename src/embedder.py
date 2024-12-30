import os
import voyageai
import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def clean_document_name(file_name):
    return file_name.replace(".txt", "").replace("_", " ")

def process_text_files_in_directory(text_dir):
    all_data = []
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    
    for text_file in text_files:
        text_file_path = os.path.join(text_dir, text_file)
        loader = TextLoader(file_path=text_file_path, encoding="utf-8")
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_data = text_splitter.split_documents(data)
        cleaned_name = clean_document_name(text_file)
        
        for document in split_data:
            document.metadata['document_name'] = cleaned_name
            all_data.append(document)
    
    return all_data

def embed_documents(page_contents, batch_size=128):
    vo = voyageai.Client()
    documents_embeddings = []
    
    for i in range(0, len(page_contents), batch_size):
        batch = page_contents[i: i + batch_size]
        embeddings = vo.embed(batch, model="voyage-3", input_type="document").embeddings
        documents_embeddings.extend(embeddings)
    
    return documents_embeddings

def save_embeddings(file_path, document_names, documents, embeddings):
    data_to_save = {
        "document_names": document_names,
        "documents": documents,
        "embeddings": embeddings
    }
    with open(file_path, "w") as f:
        json.dump(data_to_save, f)
    print(f"Embeddings saved to {file_path}")


# Initialize global variables with None
document_names = None
documents = None
documents_embeddings = None

def load_embeddings(file_path = "./data/embeddings.json"):
    """Initialize the global data from embeddings file"""
    global document_names, documents, documents_embeddings
    
    with open(file_path, "r") as f:
        data_loaded = json.load(f)
    
    document_names = data_loaded["document_names"]
    documents = data_loaded["documents"]
    documents_embeddings = data_loaded["embeddings"]
    print("Vector library loaded successfully")