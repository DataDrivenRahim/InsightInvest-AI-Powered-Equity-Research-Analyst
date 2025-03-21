#Service 3
 
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

app = Flask(__name__)

# Set Google API Key (Replace with your actual API key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBKDVPPmYIvTH1SiNdvGxHs_jUFpxAQKKA"

# Predefined Queries and Headings
query_headings = {
    "Give a short overview of the annual report provided, like the name and business of the company.": "Overview of Company",
    "What is the Revenue / Total sales of the company over Previous Years show not in tabular form ? Also provide Short Analysis that Revenue is incresing or decreasing and how this affect Investors ?": "Revenue Analysis",
    "What is the COCS (Cost of good sold ) of the company over previous years show not in tabular form ?If COCS (Cost of good sold ) is not present Calculate if (Formula GOGS=REVENUE-GROSS PROFIT) . Also provide Short Analysis that the COCS (Cost of good sold ) is increasing or decreasing and how increase and Decrease in COCS (Cost of good sold ) affect Investors": "COGS Analysis",
    "What is the Gross profit of the company over previous years show not in tabular form ? Also provide Short Analysis that the Gross Profit is increasing or decreasing and how increase and Decrease in Gross profit this affect Investors": "Gross Profit Analysis",
    "What is the Total Operating Expence of the company over previous years show not in tabular form? If Operating Expence not Present Add All Expences Except Tax and Intrest expence. Also provide Short Analysis that the Operating Expence is increasing or decreasing and how increase and Decrease in Total Operating Expence effect the investors": "Total Operating Expense Analysis",
    "What is the Operating Profit of the company over previous years show not in tabular form? Also provide Short Analysis that the Operating Profit is increasing or decreasing and how increase and Decrease in Operating Profit effect the investors": "Operating Profit Analysis",
    "What is the Net Profit of the company over previous years show not in tabular form? Also provide Short Analysis that the net profit is increasing or decreasing and how increase and Decrease in Net Profit effect the investors": "Net Profit Analysis",
    "What is the Current Ratio and Quick / acid test ratio of the company over previous years show not in tabular form? Also provide Short Analysis that the Current Ratio and Quick Ratio is increasing or decreasing and how increase and Decrease in Current Ratio and Quick Ratio effect the investors": "Current Ratio and Quick / Acid Test Ratio Analysis",
    "What is the Debt-to-Equity Ratio and Interest Coverage Ratio over previous years show not in tabular form ? Also provide Short Analysis that the Debt-to-Equity Ratio and Interest Coverage Ratio is increasing or decreasing and how increase and Decrease in Debt-to-Equity Ratio and Interest Coverage Ratio effect the investors": "Debt-to-Equity Ratio and Interest Coverage Ratio Analysis",
    "What is the Net Profit Margin, ROE, ROA, and EPS trends over previous years show not in tabular form ? , if not present calculate it? Also provide Short Analysis that the Net Profit Margin, ROE, ROA, and EPS is increasing or decreasing and how increase and Decrease inNet Profit Margin, ROE, ROA, and EPS effect the investors": "Net Profit Margin, ROE, ROA, and EPS Analysis",
    "What are the major cash flow risks identified in the report?": "Major Cash Flow Risks",
    "What are the key operational risks mentioned in the report?": "Operational Risks",
    "What are the major market risks affecting the company?": "Market Risks",
    "What are the key governance and regulatory risks mentioned in the report?": "Governance and Regulatory Risks"
}

def process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        embedding_model = GoogleGenerativeAIEmbeddings(
            api_key=os.environ["GOOGLE_API_KEY"],
            model="models/embedding-001"
        )

        semantic_splitter = SemanticChunker(embedding_model)
        chunks = semantic_splitter.split_documents(docs)

        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local("faiss_index")

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)

        system_prompt = """
                     Act as a Finance Assistant. Give answers to questions from the provided annual report. You can use images, text, and tables to answer the query. If something is not present in the annual report, just say "not present".

                    {context}
                """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def format_answer(answer):
    # Convert headings from **** to <h3>
    formatted_answer = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', answer)
    return formatted_answer

@app.route('/', methods=['GET', 'POST'])
def index():
    answers = {}
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return render_template('index.html', error='No PDF file provided.')

        pdf_file = request.files['pdf']

        if pdf_file.filename == '':
            return render_template('index.html', error='No selected file.')

        if pdf_file and pdf_file.filename.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                pdf_file.save(temp_pdf.name)
                temp_pdf_path = temp_pdf.name

            rag_chain = process_pdf(temp_pdf_path)

            if rag_chain:
                for query, heading in query_headings.items():
                    response = rag_chain.invoke({"input": query})
                    answers[heading] = format_answer(response["answer"])
            else:
                return render_template('Risk.html', error='Failed to process PDF.')

            os.unlink(temp_pdf_path)

        else:
            return render_template('Risk.html', error='Invalid file type. Please upload a PDF file.')

    return render_template('Risk.html', answers=answers)

if __name__ == '__main__':
    app.run(debug=True)





