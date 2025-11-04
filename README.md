# Guide to Tutorino

The **Tutorino** is a versatile application designed to explore and utilize **Retrieval-Augmented Generation (RAG)**. It features configurable tools for chat and a dedicated interface for managing vector databases.

---

## ⚙️ Prerequisites

To run the RAG Tutor, you'll need **Python 3** and **Ollama**.

### 1. Install Ollama

Install **Ollama** following the instructions in the [Ollama download](https://ollama.com/download).

After installation, you must download the **language and embedding models** referenced in the `chat_models` and `embedding_models` arrays within the application's source code "streamlit_unito".

### 2. Install Python Dependencies

Install the required Python packages using `pip3`. It's highly recommended to use a virtual environment.

```bash
pip3 install streamlit
pip3 install rake-nltk
pip3 install langchain
pip3 install langchain-community
pip3 install chromadb
pip3 install rank_bm25
pip3 install langchain-experimental
pip3 install langchain_ollama
pip3 install langchain_chroma
```

### 3. Download NLTK Data

The application uses specific datasets from the **Natural Language Toolkit (NLTK)**. Run the Python interpreter and execute the following commands to download the necessary data:

```python
python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
# exit()
```


---

## ▶️ Run the Application

Once all dependencies are installed, you can start the RAG Tutor using **Streamlit**.

Navigate to the project directory and run:

```bash
streamlit run streamilt_unito
```

---

## 📖 Usage

The application features two primary interfaces, accessible via the sidebar: the **DB Management** page and the **Chat Interface**.

### 1. DB Management (Database Page)

This page is used to **create a new RAG database**:

1.  Select the desired **Embedding Model** (downloaded via Ollama).
2.  Upload your source documents.
3.  The system will process the data and create a new **ChromaDB** vector store.
4.  This database can then be selected and used in the Chat Interface.

### 2. Chat Interface (Chat Page)

This is a standard **AI Chat Interface** with extensive configuration options:

* **Select a Database:** Choose a database created in the DB Management page to enable RAG. If no database is selected, the chat operates purely as a standard LLM interface.
* **Select a Chat Model:** Choose the desired Large Language Model (downloaded via Ollama).
* **Configurable Tools:** Utilize various options to adjust RAG parameters, search methods (like BM25 or vector search), and other model settings for a customized experience.
