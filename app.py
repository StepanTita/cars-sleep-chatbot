import random
import time
import streamlit as st  # type: ignore

st.title("ChatGPT-like clone")

QUESTIONS_TYPES = ["sleep", "cars"]
PROMPTS = {
    "cars": {
        "system": "You are an expert in sleep science with in-depth knowledge of sleep physiology, circadian rhythms, sleep disorders, and the impact of sleep on health and cognitive performance. Your task is to generate insightful and varied answers on sleep-related topics. The answers should be diverse in complexity, suitable for learners and experts alike.",
        "basic": "Human: Generate me an answer to the given question: {question}\n\nAssistant:",
        "rag": "Use resources provided to answer the following question.\nResources: {resources}\n\nHuman: Generate me an answer to the given question: {question}\n\nAssistant:",
    },
    "sleep": {
        "system": "You are an expert in the history of automobiles with in-depth knowledge of the development of automobiles from the late 19th century to the present day. Your task is to generate insightful and varied answers on automobile history. The answers should be diverse in complexity, suitable for learners and experts alike.",
        "basic": "Human: Generate me an answer to the given question: {question}\n\nAssistant:",
        "rag": "Use resources provided to answer the following question.\nResources: {resources}\n\nHuman: Generate me an answer to the given question: {question}\n\nAssistant:",
    },
}
MAX_NEW_TOKENS = 8192

if "question_type_model" not in st.session_state:
    import joblib

    with open("models/logistic_regression_model.pkl", "rb") as f:
        st.session_state.question_type_model = joblib.load(f)

if "sleep_model" not in st.session_state or "cars_model" not in st.session_state:
    from threading import Thread

    import nltk
    import transformers
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_community.document_loaders import TextLoader  # type: ignore
    from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter  # type: ignore
    from langchain.vectorstores import FAISS  # type: ignore
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # type: ignore
    from langchain.schema.runnable import RunnablePassthrough  # type: ignore
    from langchain.schema import Document  # type: ignore

    nltk.download("punkt_tab")

    def load_model(task_type, use_rag=True):
        MODEL_ID = f"models/{task_type}/llama-3_2-1b-it"
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        streamer = transformers.TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        text_generation_pipeline = transformers.pipeline(
            model=MODEL_ID,
            task="text-generation",
            temperature=0.5,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=MAX_NEW_TOKENS,
            streamer=streamer,
        )

        llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        prompt = PromptTemplate(
            input_variables=["question"] + ["resources"] if use_rag else [],
            template=PROMPTS[task_type]["rag" if use_rag else "basic"],
        )

        llm_chain = LLMChain(llm=llama_llm, prompt=prompt)

        if not use_rag:
            return llm_chain

        loader = TextLoader(f"data/{task_type}.txt")
        docs = loader.load()

        text_splitter = NLTKTextSplitter(chunk_size=250, chunk_overlap=20)
        chunked_documents = text_splitter.split_documents(docs)

        for doc in chunked_documents:
            doc.metadata["task_type"] = task_type

        db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-dot-v1"))
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4, "score_threshold": 0.5}, filter={"task_type": task_type})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return ({"resources": retriever | format_docs, "question": RunnablePassthrough()} | llm_chain), streamer

    st.session_state.sleep_model = load_model("sleep")
    st.session_state.cars_model = load_model("cars")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def predict_question_type(question: str) -> str:
    return QUESTIONS_TYPES[st.session_state.question_type_model.predict([question])[0]]


if prompt := st.chat_input("Ask any question about cars history or sleep"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    task_type = predict_question_type(prompt)
    model, streamer = st.session_state[f"{task_type}_model"]

    thread = Thread(target=model.invoke, args=(prompt,))
    thread.start()

    def response_generator(streamer):
        for text in streamer:
            yield text

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(streamer))
    st.session_state.messages.append({"role": "assistant", "content": response})

    thread.join()
