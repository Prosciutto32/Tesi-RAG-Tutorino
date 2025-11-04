import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import OllamaLLM
from rake_nltk import Rake
from typing import List, Tuple
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
import ollama
import time
import os
from get_embedding_function import get_embedding_function


RELEVANT_CHUNKS = 10



PROMPT_TEMPLATE = """
{prompt}

this is a history of the chat with the user:
{history}

this is the context from the course material, ordered by relevance:
{context}

---

Answer the question based only on the above context, respond in english: {question}
"""
defpreprompt = "You are a tutor for the course on Computer Networks at Pisa University." \
" Your goal is to support the students during the lecture and to answer questions about the lecture by having a " \
"conversation with them. You can generate exercises for the students and correct their answers. You can only answer questions about the course. You should refuse " \
"to answer any content not part of the course. Always be friendly, and if you cannot answer a question, admit it." \
" In summary, the tutor is a powerful system that can help with various tasks and provide valuable insight and information on various topics." \
" Whether you need help with a specific question or just want to have a conversation about a particular topic, Tutor is here to help."

embedding_model = "mxbai-embed-large"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    return query_rag(query_text, "llama3.2")



def query_rag(path, query_text: str, model_type: str, preprompt = defpreprompt, chat_history=[], preprocessing = "keyword + AI", relevant_chunks = RELEVANT_CHUNKS):
    start = time.time()  # record start time
    model= get_model(model_type)
    history_string = ""
    for message in chat_history:
        # Extract the content and remove the "Sources:" part if present
        content = message.get("content", "")
        sources_keyword = "Sources:"
        if sources_keyword in content:
            # Split the content at the first occurrence of "Sources:"
            content = content.split(sources_keyword, 1)[0].strip()

        # Only add the message to history if there is any content left after stripping
        if content:
            history_string += f"{message['role']}: {content}\n"   

    print(f"ciao, {history_string}\n")
    # Prepare the DB.
   

    embedding_function = get_embedding_function(embedding_model)
    print("embedding andato\n")
    db = Chroma(persist_directory=path, embedding_function=embedding_function) 
    print("Chroma andato\n")
    if preprocessing == "keyword":
        final_results = retriver_keyword(db, query_text, relevant_chunks)
        pass
    elif preprocessing == "semantic":
        final_results = retriver_ai(db, query_text, relevant_chunks)
        pass
    elif preprocessing == "keyword + semantic":
        final_results = retriver_ai_keyword(db, query_text, relevant_chunks//2)
        pass
    

    print("Final results after filtering:")
    for doc, score in final_results:
        print(f"Content: {doc.page_content[:100]}... (Score: {score:.4f})")
        print(f"Metadata Keywords: {doc.metadata.get('keywords', 'N/A')}")
        print("-" * 20)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in final_results])



    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(prompt = preprompt, history=history_string, context=context_text, question=query_text)
    #print(prompt)
    
    
    print("invocazione modello\n")
    # model = OllamaLLM(model=model_type)
    print("domanda inviata\n")
    response = ollama.chat(
        model=model_type, 
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = response["message"]["content"]
    #response_text = model.invoke(prompt)
    print("risposta ottenuta\n")

    sources_info = []
    for doc, score in final_results:
        source_id = doc.metadata.get("id", "N/A")
        keywords = doc.metadata.get("keywords", "N/A")
        
        # We also have the score if you want to include it.
        # score = f"{score:.2f}"
        
        sources_info.append({
            "id": source_id,
            "keywords": keywords
        })

    # Prepare the formatted output string
    formatted_sources = "\n\nSources:\n"
    for source in sources_info:
        formatted_sources += f"- ID: {source['id']}\n"
        #formatted_sources += f"  Keywords: {source['keywords']}\n"
    
    end = time.time()
    return response_text, round(end - start, 2), formatted_sources

def get_model(model_type: str):
    return OllamaLLM(model=model_type)

if __name__ == "__main__":
    main()

def retriver_keyword ( db, query_text: str, relevant_chunks: int):
    print("only keyword")
    r = Rake()
    r.extract_keywords_from_text(query_text)
    query_keywords = r.get_ranked_phrases()
    print(f"Extracted keywords from query: {query_keywords}")
    if not query_keywords:
        print("No keywords extracted from the query. Cannot perform keyword-based search.")
        return []

    print("Searching for documents based on 'keywords' metadata only...")
    keyword_conditions = [{"$contains": keyword} for keyword in query_keywords]
    keyword_filter = {"$or": keyword_conditions}

    try: #ponte H

        results = db.get(
            where_document=keyword_filter
        )

        documents_with_scores = []

        if not results:
            print("No documents found with the specified keywords.")
            return []
        
        corpus = [doc.split() for doc in results['documents']]
        tokenized_query = query_text.split()
        
        if not corpus:
            print("Corpus is empty after tokenization.")
            return []

        if not tokenized_query:
            print("Tokenized query is empty, cannot score with BM25.")
            return []
        
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        for i, doc_content in enumerate(results['documents']):
            metadata = results['metadatas'][i]
            doc_obj = Document(page_content=doc_content, metadata=metadata)
            documents_with_scores.append((doc_obj, scores[i]))
            
        documents_with_scores.sort(key=lambda item: item[1], reverse=True)


        return documents_with_scores[:relevant_chunks]


    except Exception as e: # riprova con il ponte H
        print(f"Error during keyword-only search: {e}")
        return []


def retriver_ai (db, query_text: str, relevant_chunks: int):
    print("only semantic")
    return db.similarity_search_with_score(query_text, k=relevant_chunks)

def retriever_keyword_bm25(docs: List[Document]) -> BM25Retriever:
    """Creates a BM25 retriever from a list of documents."""
    return BM25Retriever.from_documents(docs)

def retriver_ai_keyword(db: Chroma, query_text: str, relevant_chunks: int) -> List[Tuple[Document, float]]:
    print("Performing hybrid search using EnsembleRetriever.")
    # Get all documents from the database
    all_docs = db.get()
    
    # Create the retrievers
    vectorstore_retriever = db.as_retriever(search_kwargs={"k": relevant_chunks})
    docs = db.similarity_search(query_text, k=len(all_docs['documents']))
    
    bm25_retriever = retriever_keyword_bm25(docs)
    bm25_retriever.k = relevant_chunks

    # 3. Create the EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever],
        weights=[0.5, 0.5]  # You can adjust the weights for each retriever
    )

    found_relevant_chunks = ensemble_retriever.get_relevant_documents(query_text)

    documents_with_scores = [(doc, 1.0) for doc in found_relevant_chunks]

    print(f"Found {len(documents_with_scores)} chunks via hybrid search.")
    return documents_with_scores