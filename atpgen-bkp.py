import streamlit as st

import tempfile
import os
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import streamlit_pydantic as sp 

from langchain_community.document_loaders import PyMuPDFLoader,UnstructuredWordDocumentLoader,Docx2txtLoader
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain.docstore.document import Document
import json
from enum import Enum
from typing import Optional
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.openai_functions import create_qa_with_structure_chain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores import Chroma
import tiktoken

# from output import ATP
# from output1 import Finaloutput

class BaseModelWithRational(BaseModel):
    rational : str = Field(description="The rational for the provided value. Refer to all relevant sections")

class YesNoEnum(str, Enum):
    YES = 'Yes'
    NO = 'No'
    UNKNOWN = 'UNKNOWN'

class YesNoFieldWithRational(BaseModelWithRational):
    value : YesNoEnum = Field(description="Value")
    
class Finaloutput(BaseModel):
    storageSize : str = Field(description="Size of storage consumption/allowed ")
    contractingVolue: str = Field(description = "Contracting volume/price adjustment mechanism/actual production volume")
    serviceLevels: str = Field(description = "SLA Service Level Agreement (RTO recovery time objectives/TTR Time to resolution/uptime)")
    entitlements: str = Field(description = "What are the entitlements mentioned in the contract?")
    changeRequests: str = Field(description = "Request for changes and professional services")
    contractDuration: str = Field(description = "Contract start and end date")
    billingSchedule: str = Field(description = "Billing Schedules")
    fees: str = Field(description = "fees mentioned in the contract")
    serviceCredits: str = Field(description="credits")
    penalties: str = Field(description="penalties")
    OutsourcingSubContracting: str = Field(description = "Clauses related to subcontracting (sub-contracting) and subprocessing (or sub-processing)/sub-contractor/sub-processor/privacy/vendor/partner/3rd party/3rd parties/third party/third parties/adequate countries/outsource/assign/assignment/transfer rights/transfer obligations")
    OutsourcingNotices: str = Field(description = "ANotice clauses that contains email or post address we have to use to send notice of condition related to point above  ")
    customerDetails: str = Field(description = "All the customer contact details mentioned in the cotract. If there is a table, provide the whole table ")
    # subContracting: 
    # subProcessing

st.set_page_config(
    page_title="Vue PACS",
    page_icon="👩‍🎓",
)

session = st.session_state

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pz-ew-aoi-np-digitaltrans-002.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "78f01f0edc74441d859f5eb346a240dc"

"""
# Vue PACS Gen tool :female-student:
### Create an Accounting Treatment Paper in a :zap: using GenAI

As first step, upload the contract to be analysed.
"""
def file_changed():
    if 'atp' in st.session_state:
        del st.session_state['atp'] #remove old atp if new file is uploaded
        del session.messages

with st.sidebar:
    st.button("Restart Conversation",on_click=session.clear,use_container_width=True)

if "memory" not in session.keys():
    session.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

uploaded_contract_docx = st.file_uploader("Choose a Docx or PDF file", type = ['docx','pdf'], on_change=file_changed, key='file_uploader_docx')

session.emb = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", openai_api_version="2023-07-01-preview",chunk_size=16)

if uploaded_contract_docx is not None:
    st.success('File uploaded')
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(uploaded_contract_docx.getbuffer())
        fp.close()
        with st.spinner('Reading contract...'):
            if ('docx' in uploaded_contract_docx.name):
                loader = Docx2txtLoader(fp.name)
                st.session_state.contract_docs = loader.load()
            if ('pdf' in uploaded_contract_docx.name):
                loader = PyMuPDFLoader(fp.name)
                st.session_state.contract_docs = loader.load()
        st.success('Finished reading contract.')

llm_4 = AzureChatOpenAI(azure_endpoint="https://pz-ew-aoi-np-digitaltrans-002.openai.azure.com/",
                         azure_deployment="gpt-4", 
                         api_key="78f01f0edc74441d859f5eb346a240dc",
                         openai_api_version="2023-12-01-preview", max_tokens=900,temperature=0)

llm_4turbo = AzureChatOpenAI(azure_endpoint="https://pz-ew-aoi-np-digitaltrans-003.openai.azure.com/",
                        azure_deployment="gpt-4-1106-turbo", 
                        api_key="4c7f060dfcc3431897c267ed1b11cbee",
                        openai_api_version="2023-12-01-preview", max_tokens=2000,temperature=0)

llm_4turbo0125 = AzureChatOpenAI(azure_endpoint="https://pz-ew-aoi-np-digitaltrans-002.openai.azure.com/",
                         azure_deployment="gpt-4-turbo-0125", 
                         api_key="78f01f0edc74441d859f5eb346a240dc",
                         openai_api_version="2024-02-15-preview", max_tokens=2000,temperature=0)


text_splitter = TokenTextSplitter(
    chunk_size=2500,
    chunk_overlap=100,
)



def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# def templater():

systemplate = """You are an expert and reading contracts and extracting specific information to help fill a checklist.
you have already extracted the fields from individual sections of a contract. 
Based on all the individual descriptions availavle, FInd the best description and ignore the rest \n{format_instructions}\nCAnswers: {input}"""

if "memory" not in session.keys():
    session.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@chain
def stuff_document(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def num_tokens_fron_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#if there is no ATP yet, but there is a new contract, generate atp
if 'atp' not in st.session_state and 'contract_docs' in st.session_state:
    with st.spinner('Analyzing contract...'):
        Texts_docs=[]
        content=[]
        metadata=[]
        parser = PydanticOutputParser(pydantic_object=Finaloutput)
        data = st.session_state.contract_docs
        texts_all = " ".join([d.page_content for d in data])
        data_list = [d.page_content for d in data]
        data_new = [Document(page_content=" ".join(d),metadata={}) for d in split_list(data_list,10)]
        texts = text_splitter.split_text(data[0].page_content)
        texts_new = text_splitter.split_documents(data)
        for val in texts_new:
            meta={}
            meta["source"]=val.metadata["source"]
            # meta["page"]=str(val.metadata["page"])
            # meta["title"]=val.metadata["title"]
            content.append(val.page_content)
            # metadata.append(meta)
        Texts_docs.extend(texts_new)


        vectordb = Chroma.from_documents(documents=Texts_docs, embedding=session.emb)
        base_retriever = vectordb.as_retriever(n=2)
        st.session_state.retriever = base_retriever

        template = PromptTemplate(
            template="""
            You are an expert at analyzing contracts and extracting information about specific entities from contracts.
            Describe clearly all the details mentioned in the contract regarding the entities provided. 
            If the answer is not available, provide N/A as the description. 
            please don`t make up an answer.\n{format_instructions}\nContract: {contract}\n""",
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt = template)
        # HumanMessagePromptTemplate.from_template("\n\nContract: {contract}")
        ])

        general_knowhow_chain_structured = (
            { 'contract': RunnablePassthrough() } |
            prompt | 
            llm_4turbo | parser
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llm_4turbo0125,
                                               chain_type="stuff",
                                               retriever=base_retriever,
                                               return_source_documents=True)
        session.qa = qa_chain

        # out = general_knowhow_chain_structured.batch([d for d in texts])
        outfinal = general_knowhow_chain_structured.invoke(texts_all)
        # parsedop = [json.dumps(o.dict()) for o in out]
        # promptout = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt = PromptTemplate(template=systemplate,
        # input_variables=[],partial_variables={"format_instructions": parser.get_format_instructions()}))])
        # chainout = ( { 'input': RunnablePassthrough() } | promptout | llm_4turbo | parser )
        # outfinal = chainout.invoke({"input":"\n".join(parsedop)})
        st.session_state.atp = outfinal
        st.success('Processed contract!')

def generate_response(question):
    output=session.qa(question)
    return output

if 'atp' in st.session_state:

    def format_name(t):
        return t.replace('_', ' ').title()

    def print_section(items, header=None, depth=0):
        with st.container(border=True):
            h = items.__class__.__name__ if header is None else header
            st.write("#" * (depth + 1), " ", format_name(h))
            "---"

            for field, description in items.__fields__.items():
                print_field(format_name(field), getattr(items, field), description, depth)

    def print_field(k, v, description, depth=0):
        

        if isinstance(v, str):
            print(k)
            # https://docs.streamlit.io/library/api-reference/layout/st.columns
            print(v)
            st.write(f"{k}: {v}")
        else:
            with st.expander(f"{k}"):
                st.write(f"{description.description}: {v}")

    print_section(st.session_state.atp)

if "messages" not in session.keys():
    session.messages = [{"role": "assistant", "content": "Hello. Welcome to Vue PACS Contract Analyzer Bot. Please type in your query."}]

# Display chat messages
for message in session.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"],unsafe_allow_html=True)


if question := st.chat_input():
    session.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

# Generate a new response if last message is not from assistant
if session.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Please wait while I fetch relevant information...."):
            # st.write(question)
            r = generate_response(question)
            print(r)
            
            # response=json.loads(r['result'])

            answer=r['result']

            # if r['source_documents']:
            #     if ',' in response['page']: 
            #         rd_more='<a href="{file_path}#page={page}" target=_blank>Read More on page {page}</a>'.format(file_path="https://itgs2022poc9715.blob.core.windows.net/bluebot/CEA_brochure_Philips_ENG.pdf",page=str(int(response['page'].split(',')[0])+1))
            #     elif '-' in response['page']:
            #         rd_more=""   
            #     else:
            #         rd_more='<a href="{file_path}#page={page}" target=_blank>Read More on page {page}</a>'.format(file_path="https://itgs2022poc9715.blob.core.windows.net/bluebot/CEA_brochure_Philips_ENG.pdf",page=str(int(response['page'])+1))

            #     answer+='\n\n\n'+rd_more

            st.markdown(answer,unsafe_allow_html=True)
        
            # if response['source']:
            #     st.button("Click here for more info",on_click=button_click)
            
            
    message = {"role": "assistant", "content": answer}
    session.messages.append(message)



