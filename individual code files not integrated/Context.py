
from flask import Flask, render_template, request, redirect, url_for, flash, session

import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    #retriever = process_pdf(file_path)
loader=PyPDFLoader(file_path)
docs=loader.load()

@app.route("/pdfRAG", methods=["GET", "POST"])
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
            return redirect(url_for("response"))
    return render_template("pdfRAG.html")


# Set Google API Key
app = Flask(__name__)
app.secret_key = "GOOGLE_API_KEY" 

# Load Google Generative AI embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=os.environ["GOOGLE_API_KEY"],
    model="models/embedding-001"  # Correct model name
)

# Initialize Semantic Chunker
semantic_splitter = SemanticChunker(embedding_model)

# Split documents semantically
chunks = semantic_splitter.split_documents(docs)

# Store chunks into FAISS
db = FAISS.from_documents(chunks, embedding_model)

# Save FAISS index for future use
db.save_local("faiss_index")

print("Chunks stored in FAISS successfully!")


# In[3]:


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})


# In[4]:



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3, max_tokens=500)


# In[5]:


system_prompt = (""""
                 Act as an Finance Assistant Give Answer of Question From Annual report Provided . You can use Images Text and Table to Answer the Query . If Something Not Present In Annual Report So just say not Present 

"""
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# In[6]:


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# In[7]:


query = """" Give Short Overview of Anual Report Provided like name and Buissnes of company  """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *RISK ASSESMENT REPORT* "+"\n\n\n")
    file.write("\n\n\t\t\t\t *Overview Of Company* "+"\n\n\n")
    file.write(response_text + "\n")
    file.write("\n\n\t\t\t\t *Income Statement Analysis* "+"\n\n\n")


print("Response appended to company_overview.txt")


# In[8]:


query = """"Q What is the Revenue / Total sales  of the company over Previous Years show not in tabular form ? Also provide Short Analysis that Revenue is incresing or decreasing  and how this affect Investors ?"""
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" "*Revenue:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[9]:


query = """" Q1 What is the COCS (Cost of good sold ) of the company over previous years show not in tabular form ?If COCS (Cost of good sold ) is not present Calculate if (Formula GOGS=REVENUE-GROSS PROFIT) . Also provide Short Analysis that the COCS (Cost of good sold ) is increasing or decreasing and how increase and Decrease in COCS (Cost of good sold ) affect Investors """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" +"*COCS (Cost of good sold ):*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[10]:


query = """" Q1 What is the Gross profit of the company over previous years show not in tabular form ? Also provide Short Analysis that the Gross Profit is increasing or decreasing and how increase and Decrease in Gross profit this affect Investors """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" + "*Gross profit:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[11]:


query = """" Q1 What is the Total Operating Expence of the company over previous years show not in tabular form? If Operating Expence not Present Add All Expences Except Tax and Intrest expence. Also provide Short Analysis that the Operating Expence is increasing or decreasing and how increase and Decrease in Total Operating Expence effect the investors """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" + "*Total Operating Expence:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[12]:


query = """" Q1 What is the Operating Profit of the company over previous years show not in tabular form? Also provide Short Analysis that the Operating Profit is increasing or decreasing and how increase and Decrease in Operating Profit effect the investors"""
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" + "*Operating Profit:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[13]:


query = """" Q1 What is the Net Profit of the company over previous years show not in tabular form? Also provide Short Analysis that the net profit is increasing or decreasing and how increase and Decrease in Net Profit effect the investors"""
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" + "*Net Profit:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[14]:


query = """" What is the Current Ratio and Quick / acid test ratio  of the company over previous years show not in tabular form? Also provide Short Analysis that the Current Ratio and Quick Ratio is increasing or decreasing and how increase and Decrease in Current Ratio and Quick Ratio
effect the investors """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t" + "*Ratio Analysis*" + "\n")
    file.write("\n\n" + "*Liquidity Ratio Analysis*" + "\n")
    file.write("\n\n" + "Current Ratio and Quick Ratio" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[15]:


query = """" What is the Debt-to-Equity Ratio and Interest Coverage Ratio over previous years show not in tabular form ? Also provide Short Analysis that the Debt-to-Equity Ratio and Interest Coverage Ratio is increasing or decreasing and how increase and Decrease in Debt-to-Equity Ratio and Interest Coverage Ratio
effect the investors """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" + "*Laverage Ratio Analysis*" + "\n")
    file.write("\n\n" + "Debt-to-Equity Ratio and Interest Coverage Ratio:" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[16]:


query = """"What is the Net Profit Margin, ROE, ROA, and EPS trends over previous years show not in tabular form ? , if not present calculate it? Also provide Short Analysis that the Net Profit Margin, ROE, ROA, and EPS is increasing or decreasing and how increase and Decrease inNet Profit Margin, ROE, ROA, and EPS
effect the investors """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n" "*Investor Profitibility Ratio*" + "\n")
    file.write("\n\n" + "Net Profit Margin, ROE, ROA, and EPS:" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[17]:


query = """"What are the major cash flow risks identified in the report? """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t" + "Major Cash Flow Risks:" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt")


# In[18]:


query = """"What are the key operational risks mentioned in the report? """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t" +"*Operational Risks*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt") 


# In[19]:


query = """"What are the major market risks affecting the company? """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t" + "*Major Market Risks*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt") 


# In[20]:


query = """"What are the key governance and regulatory risks mentioned in the report? """
response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t" + "*Governance And Regulatory Risks*:" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Response appended to company_overview.txt") 


# In[67]:


def text_to_pdf(text_file, pdf_file):
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter
    margin = 50  # Left margin
    y_position = height - 50  # Start position

    with open(text_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        wrapped_lines = simpleSplit(line.strip(), "Helvetica", 12, width - 2 * margin)
        
        for wrapped_line in wrapped_lines:
            if y_position < 50:  # If space is low, create a new page
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - 50  # Reset position

            c.drawString(margin, y_position, wrapped_line)
            y_position -= 20  # Move down for next line

    c.save()
    print(f"PDF saved as {pdf_file}")

# Convert 'company_overview.txt' to 'company_overview.pdf'
text_to_pdf("company_overview.txt", "company_overview.pdf")



# In[ ]:


#8afd555b84b84ee9a83a720ef0de8397

