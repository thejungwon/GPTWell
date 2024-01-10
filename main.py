import os
import json
import time
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import ChatOpenAI
import streamlit as st
from openai import OpenAI

# Constants
API_KEY = "YOUR-API-KEY"
BERT_MODEL_NAME = "bert-base-uncased"

# Environment Variables
os.environ["OPENAI_API_KEY"] = API_KEY

# Initialize BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BertModel.from_pretrained(BERT_MODEL_NAME)

# Initialize OpenAI Client
client = OpenAI(api_key=API_KEY)


def get_bert_embedding(text):
    """
    Get BERT embedding for the given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def load_openai_client_and_assistants(assistant_id):
    """
    Load OpenAI client and assistants.
    """
    assistant = client.beta.assistants.retrieve(assistant_id)
    thread = client.beta.threads.create()
    return assistant, thread


def wait_on_run(run, thread):
    """
    Wait for the assistant AI to process the request.
    """
    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(0.5)
    return run


def get_assistant_response(thread, assistant_id, user_input=""):
    """
    Initiate assistant AI response.
    """
    print("SENT TO", assistant_id)
    start_time = time.time()
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_input
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant_id
    )
    run = wait_on_run(run, thread)
    messages = client.beta.threads.messages.list(
        thread_id=thread.id, order="asc", after=message.id
    )
    print(f"RECEIVED FROM {assistant_id} TOOK {time.time() - start_time}")
    return messages.data[0].content[0].text.value


def get_responses_from_all_assistants(user_input, assistants):
    """
    Handle multiple assistants in parallel.
    """
    with ThreadPoolExecutor(max_workers=len(assistants)) as executor:
        futures = [
            executor.submit(
                get_assistant_response,
                assistant["thread"],
                assistant["assistant"].id,
                user_input,
            )
            for assistant in assistants
        ]
        responses = [future.result() + "\n\n\n" for future in futures]
    return responses


def summarize(prompt, text):
    """
    Summarize the responses from assistants.
    """
    chat_model = ChatOpenAI(temperature=0)
    prompt += " These are the answers that I received from each source:\n\n"
    print(prompt + text)
    summary = chat_model.predict(prompt + text)
    return summary


def init():
    """
    Initialize the assistants and embeddings.
    """
    global GPTS, doc_embeddings
    with open("initial_assistant.json", "r") as file:
        GPTS = json.load(file)
    doc_embeddings = [get_bert_embedding(gpt["metadata"]) for gpt in GPTS]
    for gpt in GPTS:
        assistant, thread = load_openai_client_and_assistants(gpt["assistant_id"])
        gpt["assistant"] = assistant
        gpt["thread"] = thread


def setup_sidebar():
    """
    Sets up the sidebar for the Streamlit app.
    """
    with st.sidebar:
        st.header("Assistants")

        with st.form("my_form", clear_on_submit=True):
            assistant_name = st.text_input("Assistant Name 👇", placeholder="Drill Bot")
            assistant_id = st.text_input(
                "Add assistants with their key 👇",
                placeholder="asst_xxxxxxxxxxxxxxx",
            )
            assistant_url = st.text_input(
                "Add assistants with their API URL 👇",
                placeholder="https://xxx.xxx.xxx",
                disabled=True,
            )
            assistant_txt = st.text_area(
                "Metadata",
                "I am an Oil and Gas specialized bot, especially, for drilling....",
            )
            submitted = st.form_submit_button("Add")

            if submitted:
                add_new_assistant(
                    assistant_name, assistant_id, assistant_url, assistant_txt
                )

        st.divider()
        st.header("Added Assistants")
        display_added_assistants()


def add_new_assistant(name, id, url, metadata):
    """
    Adds a new assistant to the GPTS list.
    """
    new_gpt = {"name": name, "assistant_id": id, "url": url, "metadata": metadata}
    assistant, thread = load_openai_client_and_assistants(id)
    new_gpt["assistant"] = assistant
    new_gpt["thread"] = thread
    GPTS.append(new_gpt)
    print(f"Added assistant: {name}")
    print(GPTS)


def display_added_assistants():
    """
    Displays buttons for each added assistant in the sidebar.
    """
    for gpt in GPTS:
        st.button(gpt["name"], type="primary", key=gpt["name"])


def handle_user_input():
    """
    Handles user input in the main chat interface.
    """
    st.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    display_chat_history()

    # React to new user input
    if prompt := st.chat_input("Ask experts!"):
        process_user_input(prompt)


def display_chat_history():
    """
    Displays the chat history in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_input(prompt):
    """
    Processes the user input, finds relevant assistants, and displays responses.
    """
    display_user_message(prompt)
    find_and_display_responses(prompt)


def display_user_message(prompt):
    """
    Displays the user's message in the chat interface.
    """
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


def find_and_display_responses(prompt):
    """
    Finds relevant assistants and displays their responses.
    """
    with st.spinner("Finding relevant assistants..."):
        top_k_assistants = find_relevant_assistants(prompt)

    with st.spinner("Asking selected assistants..."):
        local_answers = get_responses_from_all_assistants(
            user_input=prompt, assistants=top_k_assistants
        )

    with st.spinner("Summarizing the answers..."):
        response = summarize(prompt, "\n".join(local_answers))

    display_assistant_responses(response, local_answers)


def find_relevant_assistants(prompt):
    """
    Finds assistants relevant to the user's prompt.
    """
    time.sleep(2)  # Simulate processing delay
    query_embedding = get_bert_embedding(prompt)
    similarities = [
        cosine_similarity(query_embedding, doc_emb)[0][0] for doc_emb in doc_embeddings
    ]

    # Get top K documents
    K = 3
    top_k_docs = sorted(zip(GPTS, similarities), key=lambda x: x[1], reverse=True)[:K]
    return [gpt for gpt, _ in top_k_docs]


def display_assistant_responses(response, local_answers):
    """
    Displays the responses from the assistants in the Streamlit app.
    """
    with st.chat_message("assistant"):
        st.markdown(response)
        display_individual_responses(local_answers)

    st.session_state.messages.append({"role": "assistant", "content": response})


def display_individual_responses(local_answers):
    """
    Displays each assistant's individual response.
    """

    for local_answer, assistant in zip(local_answers, GPTS):
        st.caption(assistant["name"])
        st.text(local_answer)


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("GPTWell")
    setup_sidebar()
    handle_user_input()


if __name__ == "__main__":
    init()
    main()
