import os
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone as PineconeVector
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize Pinecone and embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-l6-v2")
index = pc.Index('otl-chat-bot')
text_field = "text"
vectorstore = PineconeVector(index=index, embedding=embeddings.embed_query, text_key=text_field)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Set up chat model
chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')

# Define the augment prompt function
def augment_prompt(query: str):
    results = vectorstore.similarity_search(query, k=10)
    source_knowledge = "\n".join([f"{x.page_content}\nLocal Path: {x.metadata.get('local_path', 'No local path available')}" for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

@app.route('/', methods=['GET', 'POST'])
def index():
    ai_response = ("Hi, I'm <b>OTL GenAI Buddy</b>, your Onward GenAI ChatBot! Feel free to ask your question.<br/>  Please Note - This is an AI-generated response .<br/>If you need any further assistance,You can contact your HRBP.")
    chat_history = request.form.getlist('chat_history') if request.method == 'POST' else [f"bot:{ai_response}"]

    if request.method == 'POST':
        user_input = request.form['prompt_input']
        print(user_input)
        chat_history.append(f"user: {user_input}")
        
        prompt = augment_prompt(user_input)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hi AI, how are you today?"),
            AIMessage(content="I'm great thank you. How can I help you?"),
            HumanMessage(content="I'd like to understand string theory."),
            HumanMessage(content=prompt)
        ]
        
        res = chat(messages)
        

        results = vectorstore.similarity_search(user_input, k=1)
        if results[0].page_content != 'Thank you':
            print("yes")
            for result in results:
                local_path = result.metadata.get('local_path', 'No local path available')
                chat_history.append(f"bot: {res.content} \nReference document : {local_path}")
                
        else:
            chat_history.append(f"bot: {res.content}")
        
        
        

    return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
