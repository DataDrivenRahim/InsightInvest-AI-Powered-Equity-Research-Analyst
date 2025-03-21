from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WebBaseLoader
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import re
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


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)

app.secret_key = "GOOGLE_API_KEY"  # Replace with a strong secret key
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# app.secret_key = "GOOGLE_API_KEY"  # Replace with a strong secret key
# # app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBKDVPPmYIvTH1SiNdvGxHs_jUFpxAQKKA"

# Helper functions
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def LLm_config():
#     return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

# def process_pdf(file_path):
#     loader = PyPDFLoader(file_path)
#     docs = loader.load()

#     #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     embedding_model = GoogleGenerativeAIEmbeddings(
#     api_key=os.environ["GOOGLE_API_KEY"],
#     model="models/embedding-001"  # Correct model name
# )

#     semantic_splitter = SemanticChunker(embedding_model)
#     chunks = semantic_splitter.split_documents(docs)

#     db = FAISS.from_documents(chunks, embedding_model)
#     retriever2 = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
#     return retriever2

# def create_prompt():
#     system_prompt = (
#         "You are an assistant for question-answering with Financial Annual Report. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise.\n\n{context}"
#     )
#     return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# # Routes
@app.route("/")
def homepage():
    return render_template("homepage.html")

#@app.route("/pdfRAG", methods=["GET", "POST"])
# def pdfRAG():
#     if request.method == "POST":
#         if "file" not in request.files:
#             flash("No file part", "error")
#             return redirect(request.url)
#         file = request.files["file"]
#         if file.filename == "":
#             flash("No selected file", "error")
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(file_path)
#             session["chat_history"] = []  # Clear chat history for a new PDF
#             session["filename"] = filename
#             return redirect(url_for("response"))
#     return render_template("pdfRAG.html")

# @app.route("/response", methods=["GET", "POST"])
# def response():
#     filename = session.get("filename")
#     if not filename:
#         return redirect(url_for("pdfRAG"))

#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     retriever2 = process_pdf(file_path)
#     prompt = create_prompt()
#     llm = LLm_config()
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever2, question_answer_chain)

#     if request.method == "POST":
#         question = request.form.get("question")
#         if question:
#             result = rag_chain.invoke({"input": question})
#             answer = result["answer"]

#             # Save question and answer to session chat history
#             if "chat_history" not in session:
#                 session["chat_history"] = []
#             session["chat_history"].append({"question": question, "answer": answer})
#             session.modified = True  # Mark session as modified

#     return render_template("response.html", chat_history=session.get("chat_history", []))

# #SERVICE 2


def get_historical_stock_data(ticker: str, years: int = 5, interval: str = "1d"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Drop rows where Date conversion failed
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    return df

def create_stock_graph(df, y_column, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[y_column], mode='lines', name=title, line=dict(color=color), hoverinfo='x+y'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=f'{title} (USD)', template='plotly_dark', hovermode='x unified')
    return fig.to_html(full_html=False)

def plot_moving_averages(df, ticker, short_window=50, long_window=200):
    df['SMA'] = df['Close'].rolling(window=short_window).mean()
    df['LMA'] = df['Close'].rolling(window=long_window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price', line=dict(color='#FF5733'), hoverinfo='x+y'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA'], mode='lines', name=f'{short_window}-Day SMA', line=dict(color='#33FFCE', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['LMA'], mode='lines', name=f'{long_window}-Day LMA', line=dict(color='#FFA500', dash='dot')))
    fig.update_layout(title=f'{ticker} Moving Averages Trend', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark', hovermode='x unified')
    return fig.to_html(full_html=False)

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    seq_length = 60
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(100, return_sequences=False, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, X_test, scaler, prediction_days):
    future_input = X_test[-1].reshape(1, X_test.shape[1], 1)
    future_predictions = []
    for _ in range(prediction_days):
        pred = model.predict(future_input)
        future_predictions.append(pred[0, 0])
        pred_reshaped = np.reshape(pred, (1, 1, 1))
        future_input = np.append(future_input[:, 1:, :], pred_reshaped, axis=1)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

@app.route("/lstm", methods=[ 'GET','POST'])
def lstm():
    if request.method == 'POST':
        ticker = request.form['ticker']
        prediction_days = int(request.form['prediction_days'])
        df = get_historical_stock_data(ticker)
        closing_graph = create_stock_graph(df, 'Close', 'Closing Price Trend', '#FF5733')
        opening_graph = create_stock_graph(df, 'Open', 'Opening Price Trend', '#33FFCE')
        high_graph = create_stock_graph(df, 'High', 'High Price Trend', '#FFA500')
        volume_graph = create_stock_graph(df, 'Volume', 'Volume Trend', '#FF00FF')
        moving_avg_graph = plot_moving_averages(df, ticker)

        X, y, scaler = prepare_lstm_data(df)
        X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
        y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
        model = build_lstm_model(X.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])
        
        future_predictions = predict_future_prices(model, X_test, scaler, prediction_days)
        dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=future_predictions.flatten(), mode='lines', name='Forecasted Price', line=dict(color='#FF5733')))
        fig.update_layout(title=f'{ticker} {prediction_days}-Day Stock Price Forecast', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
        forecast_graph = fig.to_html(full_html=False)
        
        forecast_df = pd.DataFrame({'Date': dates, 'Predicted Price': future_predictions.flatten()})
        return render_template('LSTM.html', closing_graph=closing_graph, opening_graph=opening_graph, high_graph=high_graph,
                               volume_graph=volume_graph, moving_avg_graph=moving_avg_graph, forecast_graph=forecast_graph,
                               forecast_table=forecast_df.to_html(classes='table table-dark'))
    return render_template('LSTM.html', closing_graph=None, opening_graph=None, high_graph=None, volume_graph=None,
                           moving_avg_graph=None, forecast_graph=None, forecast_table=None)

#SERVICE 3


# os.environ["GOOGLE_API_KEY"] = "AIzaSyBKDVPPmYIvTH1SiNdvGxHs_jUFpxAQKKA"

# # Predefined Queries and Headings
# query_headings = {
#     "Give a short overview of the annual report provided, like the name and business of the company.": "Overview of Company",
#     "What is the Revenue / Total sales of the company over Previous Years show not in tabular form ? Also provide Short Analysis that Revenue is incresing or decreasing and how this affect Investors ?": "Revenue Analysis",
#     "What is the COCS (Cost of good sold ) of the company over previous years show not in tabular form ?If COCS (Cost of good sold ) is not present Calculate if (Formula GOGS=REVENUE-GROSS PROFIT) . Also provide Short Analysis that the COCS (Cost of good sold ) is increasing or decreasing and how increase and Decrease in COCS (Cost of good sold ) affect Investors": "COGS Analysis",
#     "What is the Gross profit of the company over previous years show not in tabular form ? Also provide Short Analysis that the Gross Profit is increasing or decreasing and how increase and Decrease in Gross profit this affect Investors": "Gross Profit Analysis",
#     "What is the Total Operating Expence of the company over previous years show not in tabular form? If Operating Expence not Present Add All Expences Except Tax and Intrest expence. Also provide Short Analysis that the Operating Expence is increasing or decreasing and how increase and Decrease in Total Operating Expence effect the investors": "Total Operating Expense Analysis",
#     "What is the Operating Profit of the company over previous years show not in tabular form? Also provide Short Analysis that the Operating Profit is increasing or decreasing and how increase and Decrease in Operating Profit effect the investors": "Operating Profit Analysis",
#     "What is the Net Profit of the company over previous years show not in tabular form? Also provide Short Analysis that the net profit is increasing or decreasing and how increase and Decrease in Net Profit effect the investors": "Net Profit Analysis",
#     "What is the Current Ratio and Quick / acid test ratio of the company over previous years show not in tabular form? Also provide Short Analysis that the Current Ratio and Quick Ratio is increasing or decreasing and how increase and Decrease in Current Ratio and Quick Ratio effect the investors": "Current Ratio and Quick / Acid Test Ratio Analysis",
#     "What is the Debt-to-Equity Ratio and Interest Coverage Ratio over previous years show not in tabular form ? Also provide Short Analysis that the Debt-to-Equity Ratio and Interest Coverage Ratio is increasing or decreasing and how increase and Decrease in Debt-to-Equity Ratio and Interest Coverage Ratio effect the investors": "Debt-to-Equity Ratio and Interest Coverage Ratio Analysis",
#     "What is the Net Profit Margin, ROE, ROA, and EPS trends over previous years show not in tabular form ? , if not present calculate it? Also provide Short Analysis that the Net Profit Margin, ROE, ROA, and EPS is increasing or decreasing and how increase and Decrease inNet Profit Margin, ROE, ROA, and EPS effect the investors": "Net Profit Margin, ROE, ROA, and EPS Analysis",
#     "What are the major cash flow risks identified in the report?": "Major Cash Flow Risks",
#     "What are the key operational risks mentioned in the report?": "Operational Risks",
#     "What are the major market risks affecting the company?": "Market Risks",
#     "What are the key governance and regulatory risks mentioned in the report?": "Governance and Regulatory Risks"
# }

# def process_pdf(pdf_path):
#     try:
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()

#         embedding_model = GoogleGenerativeAIEmbeddings(
#             api_key=os.environ["GOOGLE_API_KEY"],
#             model="models/embedding-001"
#         )

#         semantic_splitter = SemanticChunker(embedding_model)
#         chunks = semantic_splitter.split_documents(docs)

#         db = FAISS.from_documents(chunks, embedding_model)
#         db.save_local("faiss_index")

#         retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)

#         system_prompt = """
#                      Act as a Finance Assistant. Give answers to questions from the provided annual report. You can use images, text, and tables to answer the query. If something is not present in the annual report, just say "not present".

#                     {context}
#                 """

#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system_prompt),
#                 ("human", "{input}"),
#             ]
#         )

#         question_answer_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#         return rag_chain

#     except Exception as e:
#         print(f"Error processing PDF: {e}")
#         return None

# def format_answer(answer):
#     # Convert headings from **** to <h3>
#     formatted_answer = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', answer)
#     return formatted_answer

# @app.route('/risk', methods=['GET', 'POST'])
# def risk():
#     answers = {}
#     if request.method == 'POST':
#         if 'pdf' not in request.files:
#             return render_template('index.html', error='No PDF file provided.')

#         pdf_file = request.files['pdf']

#         if pdf_file.filename == '':
#             return render_template('index.html', error='No selected file.')

#         if pdf_file and pdf_file.filename.endswith('.pdf'):
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
#                 pdf_file.save(temp_pdf.name)
#                 temp_pdf_path = temp_pdf.name

#             rag_chain = process_pdf(temp_pdf_path)

#             if rag_chain:
#                 for query, heading in query_headings.items():
#                     response = rag_chain.invoke({"input": query})
#                     answers[heading] = format_answer(response["answer"])
#             else:
#                 return render_template('Risk.html', error='Failed to process PDF.')

#             os.unlink(temp_pdf_path)

#         else:
#             return render_template('Risk.html', error='Invalid file type. Please upload a PDF file.')

#     return render_template('Risk.html', answers=answers)

#SERVICE 4

NEWSAPI_KEY = "c0d8b16dbbd14701aea9abc230289c7d"
GOOGLE_API_KEY = "AIzaSyBKDVPPmYIvTH1SiNdvGxHs_jUFpxAQKKA"
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCscMh5gB2AXuPNDRqXczgt6VKODYCfd8k"
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

    documents2 = [Document(page_content=text_data)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents2)
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

@app.route("/news", methods=["GET", "POST"])
def news():
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

#service 5

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

@app.route("/web", methods=["GET", "POST"])
def web():
    if request.method == "POST":
        url = request.form.get("url")
        if not url:
            flash("No URL provided", "error")
            return redirect(request.url)
        
        session["chat_history"] = []  # Clear chat history for a new URL
        session["url"] = url
        return redirect(url_for("response2"))
    return render_template("webRAG.html")

@app.route("/response2", methods=["GET", "POST"])
def response2():
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

#service 6

os.environ["GOOGLE_API_KEY"] = "AIzaSyBKDVPPmYIvTH1SiNdvGxHs_jUFpxAQKKA"

query_headings = {  # Predefined Queries
    """Compare the revenue of Company 1 and Company 2 based on matching fiscal years. please dont give responce in tabular form  .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis on the revenue trend of both companies over the matched period.    

If revenue figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Revenue Comparison and Trend Analysis",

    """Compare the Cost of Goods Sold (COGS) of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  
- Clearly state the exact years being compared and extract COGS figures for those years.  
- Provide a short analysis on the COGS trend of both companies over the matched period.  
- Offer an investment recommendation by considering trends in COGS, revenue, and overall profitability.  

If COGS figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Cost of Goods Sold (COGS) Comparison and Investment Insights" ,

    """"Compare the Total Operating Expenses of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis of the trend in Total Operating Expenses for both companies over the matched period.    
- Based on this analysis, discuss the impact of operating expenses on profitability and provide an investment recommendation.  

If Total Operating Expense figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Total Operating Expenses and Profitability Impact",

"""Compare the Operating Profit (also known as Profit from Operations) of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  
- Provide a short analysis of the Operating Profit trend of both companies over the matched period.  
- Compare the **Operating Profit  for both companies to assess operational efficiency.  
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Operating Profit figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Operating Profit Comparison and Efficiency Analysis",

"""Compare the Net Profit of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis of the Net Profit trend of both companies over the matched period.  .    
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Net Profit Trend Analysis and Investment Recommendation"  ,

 """Compare the **Current Ratio** and **Quick (Acid-Test) Ratio** of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.   
- Provide a short analysis of the **liquidity position** of both companies over the matched period.   
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If the required ratio figures for either company are missing for any years, state that clearly instead of making assumptions.""" :"Liquidity Position: Current Ratio & Quick Ratio Analysis" ,

"""Compare the **Debt-to-Equity Ratio** and **Interest Coverage Ratio** of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis of the **financial leverage** and **debt servicing ability** of both companies over the matched period.    
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If the required ratio figures for either company are missing for any years, state that clearly instead of making assumptions.""":"Financial Leverage: Debt-to-Equity & Interest Coverage Ratio"  ,

"""Compare the **Net Profit Margin** trends of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  .  
- If Net Profit Margin is not explicitly available, calculate it using **(Net Profit / Revenue) × 100**.  
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit Margin figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Net Profit Margin Trends and Investment Implications" ,

"""Compare the **EPS** trends of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  .    
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit Margin figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Earnings Per Share (EPS) Trends and Investment Outlook" ,

"""Compare the ROE trends of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  .  

- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit Margin figures for the required years are missing for either company, state that clearly instead of making assumptions.""":"Return on Equity (ROE) Comparison and Investor Insights",

 """Compare the **major cash flow risks** of Company 1 and Company 2 based on their financial reports. Not In tabular form  """:"Cash Flow Risk Analysis for Both Companies",

 """Compare the **major market risks** affecting Company 1 and Company 2 based on their industry and financial reports. please dont give responce in tabular form""":"Market Risk Factors Affecting Both Companies",

 """Compare the **Governance and Regulatory Risks** affecting Company 1 and Company 2 based on their financial reports and industry compliance standards.please dont give responce in tabular form  """:"Governance and Regulatory Risks Analysis"

}

def process_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        embedding_model = GoogleGenerativeAIEmbeddings(api_key=os.environ["GOOGLE_API_KEY"], model="models/embedding-001")
        db = FAISS.from_texts(chunks, embedding_model)
        db.save_local("faiss_index")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)
        system_prompt = """
            Act as a Finance Assistant. Answer questions based on the uploaded text file. If data is missing, reply with 'not present'.
            {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    except Exception as e:
        print(f"Error processing TXT: {e}")
        return None

def format_answer(answer):
    return re.sub(r'\\(.?)\\*', r'<h3>\1</h3>', answer)

@app.route('/compi', methods=['GET', 'POST'])
def compi():
    answers = {}
    if request.method == 'POST':
        if 'txt' not in request.files:
            return render_template('Compi.html', error='No TXT file provided.')

        txt_file = request.files['txt']
        if txt_file.filename == '' or not txt_file.filename.endswith('.txt'):
            return render_template('Compi.html', error='Invalid file type. Please upload a TXT file.')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_txt:
            txt_file.save(temp_txt.name)
            temp_txt_path = temp_txt.name

        rag_chain = process_txt(temp_txt_path)

        if rag_chain:
            for query, heading in query_headings.items():
                response = rag_chain.invoke({"input": query})
                answers[heading] = format_answer(response["answer"])
        else:
            return render_template('Compi.html', error='Failed to process TXT file.')

        os.unlink(temp_txt_path)
    
    return render_template('Compi.html', answers=answers)




# Helper functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def LLm_config():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
@app.route("/pdfRAG", methods=["GET", "POST"])
def pdfRAG():
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
            return redirect(url_for("response"))
    return render_template("pdfRAG.html")

@app.route("/response", methods=["GET", "POST"])
def response():
    filename = session.get("filename")
    if not filename:
        return redirect(url_for("pdfRAG"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    retriever = process_pdf(file_path)
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


if __name__ == '__main__':
    #os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)



