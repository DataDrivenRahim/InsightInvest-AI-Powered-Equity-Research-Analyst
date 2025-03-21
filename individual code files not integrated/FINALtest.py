from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key
app.config["UPLOAD_FOLDER"] = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

# Helper functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def LLm_config():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

def process_pdf(file_path):
    # Load content from the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Initialize embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="models/embedding-001"
    )

    # Split the content into chunks
    semantic_splitter = SemanticChunker(embedding_model)
    chunks = semantic_splitter.split_documents(docs)

    # Create a vector store and retriever
    db = FAISS.from_documents(chunks, embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

def create_prompt():
    system_prompt = (
        "You are an assistant for question-answering with Financial Annual Reports. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Predefined Queries
PREDEFINED_QUERIES = [
    {
        "query": "Give Short Overview of Annual Report Provided like name and Business of company",
        "section_title": "*Overview Of Company*"
    },
    {
        "query": "What is the Revenue / Total sales of the company over Previous Years? Show not in tabular form. Also provide Short Analysis that Revenue is increasing or decreasing and how this affects Investors?",
        "section_title": "*Revenue:*"
    }
]

def save_response_to_file(response_text, section_title):
    """Save the response to a text file with a section title."""
    with open("company_overview.txt", "a", encoding="utf-8") as file:
        file.write("\n\n\t\t\t\t *RISK ASSESSMENT REPORT* \n\n\n")
        file.write(f"\n\n\t\t\t\t {section_title} \n\n\n")
        file.write(response_text + "\n")
        file.write("\n\n\t\t\t\t *Income Statement Analysis* \n\n\n")

# Routes
@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/webRAG", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            session["chat_history"] = []  # Clear chat history for a new PDF
            session["filename"] = filename
            session["predefined_query_index"] = 0  # Start with the first predefined query
            # Clear the text file at the start of a new session
            open("company_overview.txt", "w").close()
            print(f"File uploaded: {filename}")  # Debug log
            return redirect(url_for("response"))
    return render_template("index1.html")

@app.route("/response", methods=["GET", "POST"])
def response():
    filename = session.get("filename")
    if not filename:
        print("No filename in session. Redirecting to index.")  # Debug log
        return redirect(url_for("index"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    print(f"Processing file: {file_path}")  # Debug log

    # Process the PDF and create a retriever
    retriever = process_pdf(file_path)
    prompt = create_prompt()
    llm = LLm_config()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get the current query index
    predefined_query_index = session.get("predefined_query_index", 0)
    print(f"Current query index: {predefined_query_index}")  # Debug log

    # Process the current query
    if predefined_query_index < len(PREDEFINED_QUERIES):
        query_data = PREDEFINED_QUERIES[predefined_query_index]
        query = query_data["query"]
        section_title = query_data["section_title"]

        # Execute the predefined query
        result = rag_chain.invoke({"input": query})
        response_text = result["answer"]

        # Save the response to a text file
        save_response_to_file(response_text, section_title)

        # Add the predefined query and answer to chat history
        if "chat_history" not in session:
            session["chat_history"] = []
        session["chat_history"].append({"question": query, "answer": response_text})
        session.modified = True

        # Move to the next predefined query
        session["predefined_query_index"] = predefined_query_index + 1
        print(f"Processed query: {query}")  # Debug log

        # Redirect to the response page to process the next query
        return redirect(url_for("response"))

    # If all queries are processed, enable the download option
    all_queries_processed = predefined_query_index >= len(PREDEFINED_QUERIES)
    print(f"All queries processed: {all_queries_processed}")  # Debug log

    return render_template("response1.html", chat_history=session.get("chat_history", []), all_queries_processed=all_queries_processed)

@app.route("/download")
def download():
    # Provide the text file for download
    return send_file("company_overview.txt", as_attachment=True)

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)