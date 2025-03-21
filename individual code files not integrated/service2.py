from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
from langchain_community.document_loaders import WebBaseLoader  # Replace PyPDFLoader with WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}  # This can be removed or modified if not needed

app = Flask(__name__)
app.secret_key = "GOOGLE_API_KEY"  # Replace with a strong secret key
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Helper functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def LLm_config():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

def process_web_content(url):
    loader = WebBaseLoader(url)  # Use WebBaseLoader to load content from the URL
    docs = loader.load()

    embedding_model = GoogleGenerativeAIEmbeddings(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="models/embedding-001"  # Correct model name
    )

    semantic_splitter = SemanticChunker(embedding_model)
    chunks = semantic_splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

def create_prompt():
    system_prompt = (
        "You are an assistant for question-answering with Financial Annual Report. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Routes
@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/webRAG", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")  # Get the URL from the form
        if not url:
            flash("No URL provided", "error")
            return redirect(request.url)
        
        session["chat_history"] = []  # Clear chat history for a new URL
        session["url"] = url
        return redirect(url_for("response"))
    return render_template("index.html")  # Update the template to accept a URL instead of a file

@app.route("/response", methods=["GET", "POST"])
def response():
    url = session.get("url")
    if not url:
        return redirect(url_for("index"))

    retriever = process_web_content(url)
    prompt = create_prompt()
    llm = LLm_config()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    if request.method == "POST":
        question = request.form.get("question")
        if question:
            result = rag_chain.invoke({"input": question})
            answer = result["answer"]

            # Save question and answer to session chat history
            if "chat_history" not in session:
                session["chat_history"] = []
            session["chat_history"].append({"question": question, "answer": answer})
            session.modified = True  # Mark session as modified

    return render_template("response.html", chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)