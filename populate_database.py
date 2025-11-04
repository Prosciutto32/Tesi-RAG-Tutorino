import argparse
import os
from pathlib import Path
import shutil
import threading
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.schema.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from rake_nltk import Rake


def create_db(data_path, chroma_path):
    directory = Path(data_path)
    directory.mkdir(parents=True, exist_ok=True)
    os.makedirs(chroma_path, exist_ok=True)
    print(f"✅ New Chroma database directory created at {chroma_path}.")


def main(model = None, folder= None, progress_callback=None):
    chroma_path = None
    data_path = None
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    if folder: 
        data_path = folder
        chroma_path = folder + "_chroma"
    # Create (or update) the data store.
    create_db(data_path, chroma_path)
    chunks = split_documents(data_path, model, progress_callback)
    add_to_chroma(chunks, chroma_path, data_path, progress_callback, model)


def load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def split_documents(data_path, model, progress_callback):
    docs = load_documents(data_path)  # load PDFs as LangChain Documents
    text_splitter = SemanticChunker(
        get_embedding_function(model),
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=90
    )
    print("splitting documents")
    document_chunks = [] 
    for i, doc in enumerate(docs):
        text_chunks = text_splitter.split_text(doc.page_content)
        for chunk_text in text_chunks:
            new_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
            document_chunks.append(new_doc)
            
        progress_callback(i+1, len(docs), "split")

    print("splitting completed")
    r = Rake()
    
    # Extract keywords for each chunk and add to metadata
    i=0
    for chunk in document_chunks:
        print(f"chunk {i} is being processed to extract keywords")
        r.extract_keywords_from_text(chunk.page_content)
        keywords = r.get_ranked_phrases()
        chunk.metadata["keywords"] =  ", ".join(keywords[:5])
        i+=1

    return document_chunks


def add_documents(inizio, lavoro, chunks, chunks_ids, db, thread_safe_callback=None):
    batch_size = 10  # Increased batch size for efficiency
    for i in range(0, lavoro, batch_size):
        # Ensure the last batch doesn't exceed the length of the list
        batch_chunks = chunks[inizio + i: min(inizio + i + batch_size, inizio + lavoro)]
        batch_ids = chunks_ids[inizio + i: min(inizio + i + batch_size, inizio + lavoro)]
        db.add_documents(batch_chunks, ids=batch_ids)
        if thread_safe_callback:
            thread_safe_callback(len(batch_chunks))  # Update progress bar
        print(f"✅ Documents {inizio + i} - {min(inizio + i + batch_size, inizio + lavoro)} inserted.")


#implementare questo con diversi thread?
def add_to_chroma(chunks: list[Document], chroma_path, data_path, progress_callback, model):
    # Load the existing database.
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
        print(f"✅ Created new Chroma database directory at {chroma_path}.")
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function(model)
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        
        # Progress tracking
        processed_docs = 0
        total_docs = len(new_chunks)+1
        lock = threading.Lock()
        progress_callback(0, total_docs, "insert")
        def thread_safe_callback(batch_count):
            nonlocal processed_docs
            with lock:
                processed_docs += batch_count
                if progress_callback:
                    progress_callback(processed_docs, total_docs, "insert")
        threads = []
        num_threads = 4
        inizio = 0
        passo =(int) (len(new_chunks)/num_threads)
        for i in (range(num_threads-1)):
            thread = threading.Thread(target=add_documents, args=(inizio, passo, new_chunks, new_chunk_ids, db, thread_safe_callback))
            threads.append(thread)  # Store thread in the list
            thread.start()
            inizio+= passo
        thread = threading.Thread(target=add_documents, args=(inizio, len(new_chunks)-inizio, new_chunks, new_chunk_ids, db, thread_safe_callback))
        threads.append(thread)  # Store thread in the list
        thread.start()
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        print("documenti aggiunti") 
        if progress_callback:
            progress_callback(-1, -1, "")
    else:
        progress_callback(-1, -1, "")
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like 
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"page:{current_page_id}; chunk:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_chroma_database(chroma_path):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"✅ Chroma database directory at {chroma_path} has been deleted.")


def clear_database( data_path):
    clear_chroma_database(data_path + "_chroma")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    