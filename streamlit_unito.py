import csv
from io import StringIO
import os
import queue
import threading
import streamlit as st
from pathlib import Path
from populate_database import clear_database
from populate_database import main as populate
from query_data import query_rag 

chat_models = ["llama3.2","cogito:3b","qwen2.5:3b","granite3.3", "all-questions"]
embedding_models = ["mxbai-embed-large","ryanshillington/Qwen3-Embedding-0.6B","qwen3-embedding:8b ","embeddinggemma"]



def reset_preprompt_callback():
    st.session_state.current_preprompt = st.session_state.default_preprompt

if "default_preprompt" not in st.session_state:
    st.session_state.default_preprompt = "You are a tutor for the course on Computer Networks at Pisa University." \
        " Your goal is to support the students during the lecture and to answer questions about the lecture by having a " \
        "conversation with them. You can generate exercises for the students and correct their answers. You can only answer questions about the course. You should refuse " \
        "to answer any content not part of the course. Always be friendly, and if you cannot answer a question, admit it." \
        " In summary, the tutor is a powerful system that can help with various tasks and provide valuable insight and information on various topics." \
        " Whether you need help with a specific question or just want to have a conversation about a particular topic, Tutor is here to help."
    

if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = "mxbai-embed-large"

# Streamlit App Title
st.title("Tutorino")

# Sidebar for navigation
app_mode = st.sidebar.radio("Choose section", ["RAG Interface", "Database Management"])
@st.dialog("Create Database")
def database_form():
    dir_name = st.text_input("Database Name")
    st.selectbox(
        "selezionare la funzione di embedding che si desidera utilizzare",
        embedding_models,
        key="embedding_function"
    )
    if st.button("Submit"):
        if not os.path.exists("data/"+dir_name):
            os.makedirs("data/"+dir_name)
            st.success(f"Directory '{dir_name}' created successfully!")
            # qualcosa che crei una pagina streamlit che carica una cartella di materiale, magari implementiamola prima
        else:
            st.info(f"Directory '{dir_name}' already exists.")
        st.rerun()

if app_mode == "RAG Interface":
    # Initialize the chat history if it's not already in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Pre-prompt input in the sidebar
    preprompt = st.sidebar.text_area(
        "Write the prompt you want here",
        value=st.session_state.default_preprompt,
        key="current_preprompt"
    )
    st.sidebar.button("Reset Prompt", on_click=reset_preprompt_callback)

    # Model selection in the sidebar
    model_type = st.sidebar.selectbox(
        "Pick the model you want to use",
        chat_models
    )
    save_log = st.sidebar.checkbox("Save log to CSV", value=True)
    history_length = st.sidebar.slider("Number of history messages to use", min_value=2, max_value=7, value=4)

    directories = [
        d for d in os.listdir("data")
        if os.path.isdir("data/"+d) and (d.endswith("_chroma") )
    ]

    query_folder = st.sidebar.selectbox("choose the database you prefer", directories)

    search_method = st.sidebar.selectbox("choose the retrieval method you like",
                                        ["keyword", "semantic", "keyword + semantic"] )
    num_chunks = st.sidebar.slider("Number of relevant documents to retrieve", min_value=2, max_value=20, value=10)

    # User input for prompt
    if question := st.chat_input("Ask your question"):
        # Aggiungi la domanda dell'utente alla cronologia
        st.session_state.messages.append({"role": "user", "content": question})

        # Visualizza la domanda
        with st.chat_message("user"):
            st.markdown(question)

        if len(st.session_state.messages) > history_length*2:
            history_for_rag = st.session_state.messages[-history_length*2:]
        else:
            if len(st.session_state.messages) < 2:
                history_for_rag =[]
            else:
                history_for_rag = st.session_state.messages
        CHAT_MODEL ="qwen2.5:3b"  
        DEFAULT_VOTE = 0
        NCHUNK = 10
        questionnum = 0
        LOG_FILE_PATH = "Risposte_ottenute.csv"   
        if  model_type == "all-questions":
            output = StringIO()
            writer = csv.writer(output, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        # 1. Write Header
            header = [
                "Question Number",
                "Query Type",
                "Embedd Model",
                "Chat Model",
                "Response",
                "Response_time",
                "Sources",
                "My Vote",
                "Gemini Vote",
                "Paganelli/Forti Vote"
            ]
            writer.writerow(header)
            for question in [
                "Describe the main differences between the TCP and UDP transport protocols, including appropriate examples of their use.",
                "Describe the TCP three-way handshake; explain which segments are exchanged between the client and server to bring them both to the established connection state (ESTABLISHED)",
                "Briefly describe the AIMD congestion control mechanism during the Congestion Avoidance phase implemented in TCP Reno and explain how this mechanism is combined with flow control to determine the actual amount of data the sender can transmit.",
                "Compare GBN (Go-Back-N), SR (Selective Repeat), and TCP (without delayed ACKs). Assume that the timeout values for all three protocols are sufficiently large such that five consecutive data segments and their corresponding ACKs can be received (if not lost in the channel) by the receiving host (Host B) and the sending host (Host A) respectively. Suppose Host A sends five data segments to Host B and the second segment (sent by A) is lost. In the end, all 5 data segments have been correctly received by Host B.(a) How many segments did Host A send in total and how many ACKs did Host B send in total? What are their sequence numbers? Answer these questions for all three protocols. (b) If the timeout values for all three protocols are much greater than 5 RTT, which protocol will successfully deliver all five data segments in the shortest time interval?",
                "A client C has established a TCP connection with a web server S to download a web page consisting of three objects. At time t, immediately after sending the request for the third object, Host C sends Host S a segment with the FIN=1 flag.Considering TCP Reno, also indicate the value of the congestion window and threshold at each RTT, and calculate the time in seconds required to transfer the file under the two assumptions:a) The transmission occurs without losses.b) Segment no. 37 is lost, the subsequently transmitted segments are successfully transferred.Assume t_0 =0 ms is the instant when the file transfer begins (the first interval is therefore 0-1 RTT).",
                "Answer true or false and justify your answer with an example. In SR, can the sender receive an ACK related to a packet that falls outside its current window? In GBN, can the sender receive an ACK related to a packet that falls outside its current window? The Stop-and-Wait protocol is the same as the SR protocol with a window size equal to 1. The Stop-and-Wait protocol is the same as the GBN protocol with a window size equal to 1. 3 bytes. When TCP is in the CA phase, approximate the value of the congestion window by considering a linear increase of the window for each RTT. Assume the connection has just been established and neglect the time required for the handshake. For simplicity, assume the receive window value (Rwnd) is constant over time.",
                "What are the key functionalities of the Transport Layer in the OSI model, and how do they ensure efficient data delivery?",
                "Explain the purpose of TCP's Sliding Window mechanism and how it differs from the Selective Repeat protocol.",
                "Describe the key fields and their purpose in the UDP header and explain why the pseudo-header is necessary for the UDP checksum calculation.",
                "Explain the concepts of transport-layer multiplexing and demultiplexing, indicating what information is used to perform demultiplexing in UDP and TCP."
                ]:

                questionnum += 1
                
                testo_qwen_keyword_model, time_qwen_keyword_model, sources_qwen_keyword_model = query_rag("data/"+"Transport_Qwen_8b_chroma", question, CHAT_MODEL, preprompt,[], "keyword",NCHUNK )
                testo_qwen_semantic_model, time_qwen_semantic_model, sources_qwen_semantic_model = query_rag("data/"+"Transport_Qwen_8b_chroma", question, CHAT_MODEL, preprompt,[], "semantic",NCHUNK )
                testo_qwen_semantic_keyword_model, time_qwen_semantic_keyword_model, sources_qwen_semantic_keyword_model = query_rag("data/"+"Transport_Qwen_8b_chroma", question, CHAT_MODEL, preprompt,[], "keyword + semantic",NCHUNK )
                print("Qwen ended")

                testo_mxbai_keyword_model, time_mxbai_keyword_model, sources_mxbai_keyword_model = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", question, CHAT_MODEL, preprompt,[], "keyword",NCHUNK )
                testo_mxbai_semantic_model, time_mxbai_semantic_model, sources_mxbai_semantic_model = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", question, CHAT_MODEL, preprompt,[], "semantic",NCHUNK )
                testo_mxbai_semantic_keyword_model, time_mxbai_semantic_keyword_model, sources_mxbai_semantic_keyword_model = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", question, CHAT_MODEL, preprompt,[], "keyword + semantic",NCHUNK )
                print("Mxbai ended")

                testo_gemma_keyword_model, time_gemma_keyword_model, sources_gemma_keyword_model = query_rag("data/"+"Transport_Mixed_Gemma_chroma", question, CHAT_MODEL, preprompt,[], "keyword",NCHUNK )
                testo_gemma_semantic_model, time_gemma_semantic_model, sources_gemma_semantic_model = query_rag("data/"+"Transport_Mixed_Gemma_chroma", question, CHAT_MODEL, preprompt,[], "semantic",NCHUNK )
                testo_gemma_semantic_keyword_model, time_gemma_semantic_keyword_model, sources_gemma_semantic_keyword_model = query_rag("data/"+"Transport_Mixed_Gemma_chroma", question, CHAT_MODEL, preprompt,[], "keyword + semantic",NCHUNK )
                print("Gemma ended")

                log_entries = [
                    ("Qwen", "Keyword", testo_qwen_keyword_model.replace("\n", " "), time_qwen_keyword_model, sources_qwen_keyword_model.replace("\n", " ")),
                    ("Qwen", "Semantic", testo_qwen_semantic_model.replace("\n", " "), time_qwen_semantic_model, sources_qwen_semantic_model.replace("\n", " ")),
                    ("Qwen", "Semantic & Keyword", testo_qwen_semantic_keyword_model.replace("\n", " "), time_qwen_semantic_keyword_model, sources_qwen_semantic_keyword_model.replace("\n", " ")),

                    ("Mxbai", "Keyword", testo_mxbai_keyword_model.replace("\n", " "), time_mxbai_keyword_model, sources_mxbai_keyword_model.replace("\n", " ")),
                    ("Mxbai", "Semantic", testo_mxbai_semantic_model.replace("\n", " "), time_mxbai_semantic_model, sources_mxbai_semantic_model.replace("\n", " ")),
                    ("Mxbai", "Semantic & Keyword", testo_mxbai_semantic_keyword_model.replace("\n", " "), time_mxbai_semantic_keyword_model, sources_mxbai_semantic_keyword_model.replace("\n", " ")),

                    ("Gemma", "Keyword", testo_gemma_keyword_model.replace("\n", " "), time_gemma_keyword_model, sources_gemma_keyword_model.replace("\n", " ")),
                    ("Gemma", "Semantic", testo_gemma_semantic_model.replace("\n", " "), time_gemma_semantic_model, sources_gemma_semantic_model.replace("\n", " ")),
                    ("Gemma", "Semantic & Keyword", testo_gemma_semantic_keyword_model.replace("\n", " "), time_gemma_semantic_keyword_model, sources_gemma_semantic_keyword_model.replace("\n", " ")),
                ]

                # --- CSV Generation ---
                # 2. Write Data Rows
                for embed_model, query_type, response_text, response_time, response_sources in log_entries:
                    row = [
                        questionnum,
                        query_type,
                        embed_model,
                        CHAT_MODEL,
                        response_text,
                        response_time,
                        response_sources,
                        DEFAULT_VOTE,
                        DEFAULT_VOTE,
                        DEFAULT_VOTE
                    ]
                    # Writing the row ensures the 'Response' field is correctly quoted
                    # if it contains semicolons, maintaining the CSV integrity.
                    writer.writerow(row)

                # Get the final CSV string
                final_csv_content = output.getvalue().strip()

                # In a real environment, you would save this string to the file
                with open(LOG_FILE_PATH, 'w') as f:
                     f.write(final_csv_content)

                print(f"--- Generated Content for {LOG_FILE_PATH} ---")
                print(final_csv_content)
                print("---------------------------------------------")
                formatted_time_taken = f"\nTime taken: {time_qwen_keyword_model} seconds"
                risposta_qwen_keyword_model = f"{testo_qwen_keyword_model}\n{formatted_time_taken}\n{sources_qwen_keyword_model}"
                print(risposta_qwen_keyword_model)
                formatted_time_taken = f"\nTime taken: {time_qwen_semantic_model} seconds"
                risposta_qwen_semantic_model = f"{testo_qwen_semantic_model}\n{formatted_time_taken}\n{sources_qwen_semantic_model}"
                print(risposta_qwen_semantic_model)
                formatted_time_taken = f"\nTime taken: {time_qwen_semantic_keyword_model} seconds"
                risposta_qwen_semantic_keyword_model = f"{testo_qwen_semantic_keyword_model}\n{formatted_time_taken}\n{sources_qwen_semantic_keyword_model}"
                print(risposta_qwen_semantic_keyword_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Qwen - Keyword")
                with st.chat_message("assistant"):
                    st.markdown(risposta_qwen_keyword_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Qwen - Semantic")
                with st.chat_message("assistant"):
                    st.markdown(risposta_qwen_semantic_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Qwen - Semantic_Keyword")
                with st.chat_message("assistant"):
                    st.markdown(risposta_qwen_semantic_keyword_model)

                formatted_time_taken = f"\nTime taken: {time_mxbai_keyword_model} seconds"
                risposta_mxbai_keyword_model = f"{testo_mxbai_keyword_model}\n{formatted_time_taken}\n{sources_mxbai_keyword_model}"
                print(risposta_mxbai_keyword_model)
                formatted_time_taken = f"\nTime taken: {time_mxbai_semantic_model} seconds"
                risposta_mxbai_semantic_model = f"{testo_mxbai_semantic_model}\n{formatted_time_taken}\n{sources_mxbai_semantic_model}"
                print(risposta_mxbai_semantic_model)
                formatted_time_taken = f"\nTime taken: {time_mxbai_semantic_keyword_model} seconds"
                risposta_mxbai_semantic_keyword_model = f"{testo_mxbai_semantic_keyword_model}\n{formatted_time_taken}\n{sources_mxbai_semantic_keyword_model}"
                print(risposta_mxbai_semantic_keyword_model)
                formatted_time_taken = f"\nTime taken: {time_gemma_keyword_model} seconds"

                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Mxbai - Keyword")
                with st.chat_message("assistant"):
                    st.markdown(risposta_mxbai_keyword_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Mxbai - Semantic")
                with st.chat_message("assistant"):
                    st.markdown(risposta_mxbai_semantic_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Mxbai - Semantic_Keyword")
                with st.chat_message("assistant"):
                    st.markdown(risposta_mxbai_semantic_keyword_model)

                formatted_time_taken = f"\nTime taken: {time_gemma_keyword_model} seconds"
                risposta_gemma_keyword_model = f"{testo_gemma_keyword_model}\n{formatted_time_taken}\n{sources_gemma_keyword_model}"
                print(risposta_gemma_keyword_model)
                formatted_time_taken = f"\nTime taken: {time_gemma_semantic_model} seconds"
                risposta_gemma_semantic_model = f"{testo_gemma_semantic_model}\n{formatted_time_taken}\n{sources_gemma_semantic_model}"
                print(risposta_gemma_semantic_model)
                formatted_time_taken = f"\nTime taken: {time_gemma_semantic_keyword_model} seconds"
                risposta_gemma_semantic_keyword_model = f"{testo_gemma_semantic_keyword_model}\n{formatted_time_taken}\n{sources_gemma_semantic_keyword_model}"
                print(risposta_gemma_semantic_keyword_model)

                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Gemma - Keyword")
                with st.chat_message("assistant"):
                    st.markdown(risposta_gemma_keyword_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL}  - Gemma - Semantic")
                with st.chat_message("assistant"):
                    st.markdown(risposta_gemma_semantic_model)
                with st.chat_message("user"):
                    st.markdown(f"{CHAT_MODEL} - Gemma - Semantic_Keyword")
                with st.chat_message("assistant"):
                    st.markdown(risposta_gemma_semantic_keyword_model)


        else:
            with st.spinner("Processing your request, please wait..."):
                testo, tempo, sources = query_rag("data/"+query_folder, question, model_type, preprompt, history_for_rag, search_method, num_chunks)
            formatted_time_taken = f"\nTime taken: {tempo} seconds"
            risposta = f"{testo}\n{formatted_time_taken}\n{sources}"

            # if save to domanda_log.csv
            if save_log:
                LOG_FILE_PATH = "logs/domanda_log.csv"
                file_exists = os.path.isfile(LOG_FILE_PATH)
                with open(LOG_FILE_PATH, mode='a', newline='') as file:
                    writer = csv.writer(file, delimiter=';')
                    if not file_exists:
                        # Write header if file does not exist
                        writer.writerow(["Question", "Model", "DB_Used", "Search Method", "Prompt", "Response", "Time Taken", "Sources"])
                    # Write the data row separated by semicolons
                    writer.writerow([question.replace("\n", " ").replace(";", ","), model_type.replace("\n", " ").replace(";", ","), query_folder.replace("\n", " ").replace(";", ","), search_method.replace("\n", " ").replace(";", ","), preprompt.replace("\n", " ").replace(";", ","), testo.replace("\n", " ").replace(";", ","), tempo, sources.replace("\n", " ").replace(";", ",")])

            # Visualizza la risposta del modello
            with st.chat_message("assistant"):
                st.markdown(risposta)

            # Aggiungi la risposta del modello alla cronologia
            st.session_state.messages.append({"role": "assistant", "content": risposta})

elif app_mode == "Database Management":
    
    if "db_cleared" not in st.session_state:
        st.session_state.db_cleared = False
    if "db_updated" not in st.session_state:
        st.session_state.db_updated = False

    directories = [
        d for d in os.listdir("data")
        if os.path.isdir("data/"+d) and not (d.endswith("_chroma") or d.startswith("__"))
    ]

    UPLOAD_FOLDER = st.sidebar.selectbox("Select the database", directories)
    if UPLOAD_FOLDER:
        directory = Path("data/" + UPLOAD_FOLDER)
        directory.mkdir(parents=True, exist_ok=True)

    # File uploader for documents
    uploaded_files = st.file_uploader(
        "Upload document",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Upload files to the database"):
        if not uploaded_files:
            st.warning("No files selected. Please upload at least one file.")
        else:
            with st.spinner("Uploading files..."):
                for uploaded_file in uploaded_files:
                    file_path = directory / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

            # Progress bar setup
            progress_bar = st.progress(0)
            progress_text = st.empty()
            progress_queue = queue.Queue()

            def update_progress_main_thread():
                text=""
                processed = 0
                total = None
                while True:
                    if total == None:
                        progress_bar.progress(0)
                        progress_text.text(f"processing the input, this may take a feaw seconds")

                    try:
                        msg = progress_queue.get(timeout=0.2)
                        if msg == (-1, -1, ""):
                            break
                        processed, total, state = msg
                        if (state == "split"):
                            text = (f"{processed} of {total} documents splitted...")
                        if (state == "insert"):
                            text = (f"{processed} of {total} documents inserted...")
                        percent = int(processed / total * 100)
                        progress_bar.progress(percent)
                        progress_text.text(text)
                    except queue.Empty:
                        continue

            # Callback for progress
            def progress_callback(processed, total, state):
                progress_queue.put((processed, total, state))


            # Run populate in separate thread
            populate_thread = threading.Thread(
                target=populate,
                kwargs={"model":st.session_state.embedding_function, "folder": "data/" + str(UPLOAD_FOLDER), "progress_callback": progress_callback}
            )
            populate_thread.start()

            # Update progress bar
            update_progress_main_thread()

            # Set flag + rerun
            st.session_state.db_updated = True
            st.rerun()

    if st.button("Delete database"):
        clear_database("data/" + UPLOAD_FOLDER)
        st.session_state.db_cleared = True
        st.rerun()

    if st.session_state.db_updated:
        st.success("🎉 Files uploaded and database updated!")
        st.session_state.db_updated = False

    if st.session_state.db_cleared:
        st.success("🎉 Database emptied, files deleted!")
        st.session_state.db_cleared = False

    if st.sidebar.button("Create new database"):
        with st.form("new_database"):
            database_form()
