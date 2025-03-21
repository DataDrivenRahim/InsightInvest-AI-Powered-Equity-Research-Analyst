from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

NEWSAPI_KEY = "c0d8b16dbbd14701aea9abc230289c7d"
GOOGLE_API_KEY = "AIzaSyBKDVPPmYIvTH1SiNdvGxHs_jUFpxAQKKA"
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# Function to fetch news
def fetch_news(ticker):
    company_name = yf.Ticker(ticker).info.get("shortName", ticker)
    newsapi_news = newsapi.get_everything(q=company_name, language="en", sort_by="publishedAt", page_size=20)
    yahoo_news = yf.Ticker(ticker).news

    articles = []
    for article in newsapi_news.get("articles", []):
        articles.append({
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "published_at": article["publishedAt"],
            "source_type": "NewsAPI"
        })

    for article in yahoo_news:
        articles.append({
        "source": article.get("provider", "Unknown"),  # Avoid KeyError
        "title": article.get("title", "No Title Available"),  # Avoid KeyError
        "description": article.get("summary", "No Description Available"),  # Avoid KeyError
        "url": article.get("link", "#"),  # Ensure URL fallback
        "published_at": pd.to_datetime(article.get("providerPublishTime", 0), unit="s"),  # Handle missing timestamps
        "source_type": "Yahoo Finance"
    })



    df = pd.DataFrame(articles)
    df.to_csv("news.csv", index=False)
    return df

# Function to generate summary
def generate_summary():
    with open("news.csv", "r", encoding="utf-8") as file:
        text_data = file.read()

    documents = [Document(page_content=text_data)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    embedding_model = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")
    db = FAISS.from_documents(chunked_docs, embedding_model)
    db.save_local("faiss_index")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    system_prompt = "Summarize financial news in a structured format with key indicators."
    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{context}")  # 'context' is the required input variable
])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": "Summarize all news with detailed analysis."})
    return response["answer"]

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        ticker = request.form.get("ticker").upper()
        action = request.form.get("action")

        if action == "download_csv":
            fetch_news(ticker)
            return send_file("news.csv", as_attachment=True)

        elif action == "generate_summary":
            fetch_news(ticker)
            summary = generate_summary()

    return render_template("news.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
