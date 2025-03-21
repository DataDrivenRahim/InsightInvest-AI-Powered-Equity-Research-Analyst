from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker

app = Flask(__name__)
app.secret_key = "GOOGLE_API_KEY"  # Replace with a strong secret key

def LLm_config():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

def process_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="models/embedding-001"
    )
    
    db = FAISS.from_documents(docs, embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

def create_prompt():
    system_prompt = (
        "You are an assistant for question-answering based on web articles. "
        "Use the retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        if not url:
            flash("No URL provided", "error")
            return redirect(request.url)
        
        session["chat_history"] = []  # Clear chat history for a new URL
        session["url"] = url
        return redirect(url_for("response"))
    return render_template("webRAG.html")

@app.route("/response", methods=["GET", "POST"])
def response():
    url = session.get("url")
    if not url:
        return redirect(url_for("index"))

    retriever = process_url(url)
    prompt = create_prompt()
    llm = LLm_config()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    if request.method == "POST":
        question = request.form.get("question")
        if question:
            result = rag_chain.invoke({"input": question})
            answer = result["answer"]

            if "chat_history" not in session:
                session["chat_history"] = []
            session["chat_history"].append({"question": question, "answer": answer})
            session.modified = True

    return render_template("response.html", chat_history=session.get("chat_history", []))

if __name__ == '__main__':
    # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
