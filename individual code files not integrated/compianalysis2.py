# from flask import Flask, render_template, request, jsonify
# import os
# from langchain.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import tempfile
# import re

# app = Flask(_name_)

# # Set Google API Key (Replace with your actual API key)
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# query_headings = {
#    """Compare the revenue of Company 1 and Company 2 based on matching fiscal years. please dont give responce in tabular form  .  

# - If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
# - If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
# - Provide a short analysis on the revenue trend of both companies over the matched period.    

# If revenue figures for the required years are missing for either company, state that clearly instead of making assumptions.""" ,
# }

# def process_txt(txt_path):
#     try:
#         with open(txt_path, "r", encoding="utf-8") as file:
#             text_data = file.read()

#         embedding_model = GoogleGenerativeAIEmbeddings(
#             api_key=os.environ["GOOGLE_API_KEY"],
#             model="models/embedding-001"
#         )
        
#         db = FAISS.from_texts([text_data], embedding_model)
#         db.save_local("faiss_index")
#         retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)

#         system_prompt = """
#             Act as a Financial Analyst. Answer questions based on the provided text. If a question cannot be answered from the text, respond with "not present".
#             {context}
#         """
        
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "{input}"),
#         ])

#         question_answer_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#         return rag_chain
#     except Exception as e:
#         print(f"Error processing TXT file: {e}")
#         return None

# def format_answer(answer):
#     formatted_answer = re.sub(r'\\(.?)\\*', r'<h3>\1</h3>', answer)
#     return formatted_answer

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     answers = {}
#     if request.method == 'POST':
#         if 'txt' not in request.files:
#             return render_template('Compi.html', error='No TXT file provided.')

#         txt_file = request.files['txt']

#         if txt_file.filename == '':
#             return render_template('Compi.html', error='No selected file.')

#         if txt_file and txt_file.filename.endswith('.txt'):
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_txt:
#                 txt_file.save(temp_txt.name)
#                 temp_txt_path = temp_txt.name

#             rag_chain = process_txt(temp_txt_path)

#             if rag_chain:
#                 for query, heading in query_headings.items():
#                     response = rag_chain.invoke({"input": query})
#                     answers[heading] = format_answer(response["answer"])
#             else:
#                 return render_template('compi.html', error='Failed to process TXT file.')

#             os.unlink(temp_txt_path)
#         else:
#             return render_template('compi.html', error='Invalid file type. Please upload a TXT file.')

#     return render_template('compi.html', answers=answers)

# if _name_ == '_main_':
#     app.run(debug=True)




from flask import Flask, render_template, request
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

@app.route('/', methods=['GET', 'POST'])
def index():
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

if __name__ == '__main__':
    app.run(debug=True)