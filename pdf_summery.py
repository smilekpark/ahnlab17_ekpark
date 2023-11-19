
#  pdf 문서를 map_reduce_chain를 이용해서 요약해 보기 

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain


from utils import (
  BusyIndicator,
  ConsoleInput,
  get_filename_without_extension,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path,
  load_pdf
  )

llm_model = "gpt-3.5-turbo"

def summary1():

    # pdf load 
    # RecursiveCharacterTextSplitter로 분할 
    split_docs = load_pdf("./data/프리랜서 가이드라인(출판본).pdf", True)

    llm = ChatOpenAI(model_name=llm_model, temperature=0)

    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)

    summarize_document_chain.run(raw_text)


    map_template = """다음은 문서 중 일부 내용입니다
    {pages}
    이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
    요약:"""

    # Map 프롬프트 완성
    map_prompt = PromptTemplate.from_template(map_template)

    # Map에서 수행할 LLMChain 정의
    llm = ChatOpenAI(temperature=0, 
                    model_name='gpt-3.5-turbo-16k')
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce 단계에서 처리할 프롬프트 정의
    reduce_template = """ 요약 문서입니다.:
    {doc_summaries}
    이것들을 바탕으로 전체적으로 한굴로 요약해 주세요 .
    요약 :"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,                
        document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
    )

    # Map 문서를 통합하고 순차적으로 Reduce합니다.
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000, # 문서 그룹화 토큰 갯수 
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="pages",
        return_intermediate_steps=False,
    )

    result = map_reduce_chain.run(split_docs)

    print(result)


def summary2():

    llm = ChatOpenAI(model_name=llm_model, temperature=0)

    # pdf load 
    # RecursiveCharacterTextSplitter로 분할 
    split_docs = load_pdf("./data/프리랜서 가이드라인 (출판본).pdf", True)
    
 
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(split_docs)
    print(summary)
    

import fitz
def make_index():
    pdf_path = "./data/프리랜서 가이드라인 (출판본).pdf"
    toc = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        if 'ToC' in page.get_text("text"):
            outline = page.get_table_of_contents()
            toc.extend(outline)

    pdf_document.close()

    print("목차 정보:")
    for item in toc:
        print(f"제목: {item[1]}, 페이지 번호: {item[0]}, 레벨: {item[2]}")
        
    
     
if __name__ == '__main__':
    #summary1()
    # pdf 문서 요약
 #   summary2()
    
    #index 만들기 
    make_index()
