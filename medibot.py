import os
os.environ["PORT"] = os.environ.get("PORT", "10000")

import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    return llm


def main():
    st.title("🩺 Medibot - Your Simple Medical Assistant!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Ask your question here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
               Answer the user's question based on the provided context. 
If the context does not contain enough information, feel free to use your general knowledge.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID),
                 chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # If the answer indicates "I don't know", fallback to general LLM
            fallback_phrases = ["not mentioned", "don't know", "no information"]
            if any(phrase in result.lower() for phrase in fallback_phrases):
                general_llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3")
                fallback_result = general_llm.invoke(prompt)
                result = fallback_result
                source_documents = []

            # Format the result
            formatted_result = f"**🧠 Answer:**\n{result.strip()}\n\n"

            # Format the sources nicely (only if available)
            formatted_sources = "**📚 Source Documents:**\n"
            for doc in source_documents:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content.strip().split('\n')[0][:300] + "..."
                formatted_sources += f"- **Source:** `{source}` | **Page:** {page}\n  > {content}\n\n"

            result_to_show = formatted_result
            if source_documents:
                result_to_show += formatted_sources




            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


