#!/usr/bin/env python
# coding: utf-8

# In[1]:


file1 = "company_overview.txt"
file2 = "company_overview2.txt"
output_file = "combined_text.txt"

# Read content from both files
with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
    content1 = f1.read()
    content2 = f2.read()

# Write combined content to a new file
with open(output_file, "w", encoding="utf-8") as out:
    out.write(content1 + "\n" + content2)

print(f"Files concatenated successfully into {output_file}")


# In[31]:


from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os 
from langchain.vectorstores import FAISS


# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDxhPMlJLbGHMvBzXbgV8ldG9-JlReq1Hg"

# Load Google Generative AI embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=os.environ["GOOGLE_API_KEY"],
    model="models/embedding-001"  # Correct model name
)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",temperature=0.3, max_tokens=500)

# Load text file
file_path = "combined_text.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()  # Read entire content of the file

# Convert each text file into a Document object
documents = [Document(page_content=text_data)]




text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(chunked_docs, embedding_model)

# Save FAISS index for future use
db.save_local("faiss_index")

print("Chunks stored in FAISS successfully!")


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

system_prompt = (""""
                 Act As an Competitor Finance Analyst . I have Provided you some key Financial matrices and some key Risk of 2 companys , Compare the matrices of 2 Companys and give answer

"""
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# In[18]:


query = "What is revenue of Gul Ahmed ?"
results = db.similarity_search(query, k=20)  # Get the top 3 matching documents


# In[4]:


results


# In[32]:


query="""Compare the revenue of Company 1 and Company 2 based on matching fiscal years. please dont give responce in tabular form  .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis on the revenue trend of both companies over the matched period.    

If revenue figures for the required years are missing for either company, state that clearly instead of making assumptions.""" 
response = rag_chain.invoke({"input": query})
# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Competitor Analysis * "+"\n\n\n")
    file.write("\n\n\t\t\t\t *Income Statement Analysis Of Both Companys * "+"\n\n\n")
    file.write("\n\n" "*Revenue Comparison Of Both Companies :*" + "\n")  # Add query for reference
    file.write(response_text + "\n")


print("Response appended to company_overview.txt") 


# In[33]:


query = """Compare the Cost of Goods Sold (COGS) of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  
- Clearly state the exact years being compared and extract COGS figures for those years.  
- Provide a short analysis on the COGS trend of both companies over the matched period.  
- Offer an investment recommendation by considering trends in COGS, revenue, and overall profitability.  

If COGS figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *COGS (Cost of Goods Sold) Analysis Of Both Companies * "+"\n\n\n")
    file.write("\n\n" "*COGS Comparison:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("COGS analysis appended to company_overview.txt")  


# In[34]:


query = """Compare the Total Operating Expenses of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis of the trend in Total Operating Expenses for both companies over the matched period.    
- Based on this analysis, discuss the impact of operating expenses on profitability and provide an investment recommendation.  

If Total Operating Expense figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Total Operating Expense Analysis Of Both Companies* \n\n\n")
    file.write("\n\n" "*Total Operating Expense Comparison:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Total Operating Expense analysis appended to company_overview3.txt")  


# In[35]:


query = """Compare the Operating Profit (also known as Profit from Operations) of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  
- Provide a short analysis of the Operating Profit trend of both companies over the matched period.  
- Compare the **Operating Profit  for both companies to assess operational efficiency.  
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Operating Profit figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Operating Profit Analysis Of Both Companies * \n\n\n")
    file.write("\n\n" "*Operating Profit Comparison:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Operating Profit analysis appended to company_overview3.txt")  


# In[36]:


query = """Compare the Net Profit of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis of the Net Profit trend of both companies over the matched period.  .    
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Net Profit Analysis Of Both Companies * \n\n\n")
    file.write("\n\n" "*Net Profit Comparison:*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Net Profit analysis appended to company_overview3.txt")  


# In[37]:


query = """Compare the **Current Ratio** and **Quick (Acid-Test) Ratio** of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.   
- Provide a short analysis of the **liquidity position** of both companies over the matched period.   
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If the required ratio figures for either company are missing for any years, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Liquidity Ratio Analysis Of Both Companies * \n\n\n")
    file.write("\n\n\t\t\t\t *Comparison of Current Ratio and Quick Ratio of both Comoanies * \n\n\n")
    file.write("\n\n" "*Liquidity Ratio Comparison Of Both Companies :*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Liquidity Ratio analysis appended to company_overview3.txt")


# In[38]:


query = """Compare the **Debt-to-Equity Ratio** and **Interest Coverage Ratio** of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.    
- Provide a short analysis of the **financial leverage** and **debt servicing ability** of both companies over the matched period.    
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If the required ratio figures for either company are missing for any years, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Leverage Ratio Analysis Of Both Companies * \n\n\n")
    file.write("\n\n\t\t\t\t *Comparison of Debt-to-Equity Ratio and Interest Coverage Ratio Of Both Companies * \n\n\n")
    file.write("\n\n" "*Leverage Ratio Comparison Of Both Companies :*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Leverage Ratio analysis appended to company_overview3.txt")


# In[39]:


query = """Compare the **Net Profit Margin** trends of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  .  
- If Net Profit Margin is not explicitly available, calculate it using **(Net Profit / Revenue) × 100**.  
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit Margin figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Profitability Ratio Analysis Of Both Companies * \n\n\n")
    file.write("\n\n\t\t\t\t *Net Profit Margin ComparisonOf Both Companies * \n\n\n")
    file.write("\n\n" "*Net Profit Margin Comparison Of Both Companies :*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Net Profit Margin analysis appended to company_overview3.txt")


# In[40]:


query = """Compare the **EPS** trends of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  .    
- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit Margin figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Profitability Ratio Analysis Of Both Companies * \n\n\n")
    file.write("\n\n\t\t\t\t *Net Profit Margin ComparisonOf Both Companies * \n\n\n")
    file.write("\n\n" "*Net Profit Margin Comparison Of Both Companies :*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Net Profit Margin analysis appended to company_overview3.txt")


# In[41]:


query = """Compare the ROE trends of Company 1 and Company 2 based on matching fiscal years please dont give responce in tabular form .  

- If Company 1 has more years of data than Company 2, only compare the most recent matching years.  
- If Company 2 has more years of data than Company 1, limit the comparison to the available years of Company 1.  .  

- Based on this analysis, provide an investment recommendation, highlighting key insights for investors.  

If Net Profit Margin figures for the required years are missing for either company, state that clearly instead of making assumptions."""  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Profitability Ratio Analysis Of Both Companies * \n\n\n")
    file.write("\n\n\t\t\t\t *Net Profit Margin ComparisonOf Both Companies * \n\n\n")
    file.write("\n\n" "*Net Profit Margin Comparison Of Both Companies :*" + "\n")  # Add query for reference
    file.write(response_text + "\n")

print("Net Profit Margin analysis appended to company_overview3.txt")


# In[42]:


query = """Compare the **major cash flow risks** of Company 1 and Company 2 based on their financial reports. Not In tabular form  """


response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Comparison of Both Companies' Cash Flow Risks*" + "\n\n")
    file.write(response_text + "\n")
print("Cash Flow Risk comparison appended to company_overview2.txt")


# In[43]:


query = """Compare the **major market risks** affecting Company 1 and Company 2 based on their industry and financial reports. please dont give responce in tabular form"""


response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Comparison of Both Companies' Market Risks*" + "\n\n")
    file.write(response_text + "\n")
print("Market Risk comparison appended to company_overview2.txt")


# In[44]:


query = """Compare the **Governance and Regulatory Risks** affecting Company 1 and Company 2 based on their financial reports and industry compliance standards.please dont give responce in tabular form  """

  

response = rag_chain.invoke({"input": query})

# Extract the answer text
response_text = response["answer"]

# Save the response to a text file (Append mode)
with open("company_overview3.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\t\t\t\t *Comparison of Both Companies' Governance & Regulatory Risks*" + "\n\n")
    file.write(response_text + "\n")

print("Governance & Regulatory Risk comparison appended to company_overview2.txt")


# In[ ]:





# In[ ]:





# In[ ]:




