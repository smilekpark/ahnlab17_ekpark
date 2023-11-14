from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI()
chat(
  [ 
    SystemMessage(content="당신은 신혼여행 휴양지를 소개시켜주는 AI 봇입니다."),
    HumanMessage(content="1월 신혼여행지는 어디 가면 좋을까??"), 
    AIMessage(content="칸문을 추천 드립니다."), 
    HumanMessage(content="칸쿤은 이미 가봤어. 다른 곳은 없어?")
  ]
)
from langchain.schema import Document

Document(page_content="1월에 신혼여행을 계획 중이시라면, 다음과 같은 여행지를 추천해드립니다.\n\n1. 뉴질랜드의 퀸스타운(Queenstown): 야외 활동이 풍부하고 아름다운 자연 경관으로 유명한 지역입니다. 스카이 다이빙, 스카이 스윙, 스노우보드 등 다양한 액티비티를 즐길 수 있습니다.\n\n2. 태국의 코 사무이(Koh Samui): 아름다운 해변과 푸른 바다, 그리고 현지 문화를 경험할 수 있는 다양한 투어가 준비되어 있습니다. 더불어 태국 요리 체험도 추천드립니다.",
metadata={
    'my_document_id' : 234234,
    'my_document_source' : "The LangChain Papers",
    'my_document_create_time' : 1680013019
})