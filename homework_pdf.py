#!/usr/bin/env python
# coding: utf-8

# # Document Splitting


import os
import time
import json

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader


def test_pdf_split() -> None:
  loader = PyPDFLoader("./data/프리랜서 가이드라인 (출판본).pdf")
  global pages
  pages = loader.load()
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
  )
  # 분할 실행
  docs = text_splitter.split_documents(pages)

  print(f"len(docs)=>{len(docs)}")
  print(f"len(pages)=>{len(pages)}")
  return None




def test_token_split() -> None:
  text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
  text1 = "foo bar bazzyfoo"
  result = text_splitter.split_text(text1)
  print(f"result=>{result}")

  text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
  docs = text_splitter.split_documents(pages)
  print(f"docs[0]=>{docs[0]}")
  print(f"pages[0].metadata=>{pages[0].metadata}")
  return None

if __name__ == '__main__':
  test_pdf_split()
