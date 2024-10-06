import random
import time
import streamlit as st  # type: ignore

st.title("ChatGPT-like clone")

QUESTIONS_TYPES = ["sleep", "cars"]
PROMPTS = {
    "cars": {
        "system": "You are an expert in sleep science with in-depth knowledge of sleep physiology, circadian rhythms, sleep disorders, and the impact of sleep on health and cognitive performance. Your task is to generate insightful and varied answers on sleep-related topics. The answers should be diverse in complexity, suitable for learners and experts alike.",
        "rag": "Use resources provided to answer the following question.\nResources: {resources}\n\nHuman: Generate me an answer to the given question: {question}\n\nAssistant:",
    },
    "sleep": {
        "system": "You are an expert in the history of automobiles with in-depth knowledge of the development of automobiles from the late 19th century to the present day. Your task is to generate insightful and varied answers on automobile history. The answers should be diverse in complexity, suitable for learners and experts alike.",
        "rag": "Use resources provided to answer the following question.\nResources: {resources}\n\nHuman: Generate me an answer to the given question: {question}\n\nAssistant:",
    },
}
MAX_NEW_TOKENS = 8192

if "question_type_model" not in st.session_state:
    import joblib

    with open("models/logistic_regression_model.pkl", "rb") as f:
        st.session_state.question_type_model = joblib.load(f)

if "sleep_model" not in st.session_state or "cars_model" not in st.session_state:
    import transformers
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    def load_model(task_type):
        MODEL_ID = f"models/{task_type}/llama-3_2-1b-it"
        text_generation_pipeline = transformers.pipeline(
            model=MODEL_ID,
            task="text-generation",
            temperature=0.5,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        
        llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        prompt = PromptTemplate(
            input_variables=["question", "resources"],
            template=PROMPTS[task_type]['rag'],
        )

        return LLMChain(llm=llama_llm, prompt=prompt)

    st.session_state.sleep_model = load_model("sleep")
    st.session_state.cars_model = load_model("cars")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def predict_question_type(question: str) -> str:
    return QUESTIONS_TYPES[st.session_state.question_type_model.predict([question])[0]]


if prompt := st.chat_input("Ask any question about cars history or sleep"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    task_type = predict_question_type(prompt)
    model = st.session_state[f"{task_type}_model"]
    reply = model(prompt)
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
