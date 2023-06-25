
import pandas as pd 
import numpy as np 
import requests 
from bs4 import BeautifulSoup as bs
import time
import random
import os
import openai

page_no = "1"
url = f"https://sgsg.hankyung.com/sgplus/quiz?page={page_no}"

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"} #google에 user agent string 검색
quiz_link = requests.get(url, headers = headers)
quiz_link.status_code
html = bs(quiz_link.text, 'html.parser')

data = []
for p in html.find_all('p'):
    data.append(p.text)

dfs = []

# 리스트를 데이터프레임으로 변환
for a in range(0,5):
    df_a = pd.DataFrame(data, columns=['text'])
    
    df_a = pd.DataFrame(df_a['text'][2:-1])
    
    df_a['text'].iloc[a]
    df_a['text'].iloc[a].split('?')
    
    temp_a = df_a['text'].iloc[a]
    temp_a = temp_a.split('?')
    temp_a[0] + '?'
    df_a['text'].iloc[a].split("?")[1].split('.')[0][:-1]
    df_a['text'].iloc[a].split("?")[1].split('.')[1] + '?'
    df_a['text'].iloc[a].split("?")[2].split('.')[0][:-1]
    
    question = []
    answer_list = []
    temp_a = df_a['text'].iloc[a]
    question.append(temp_a.split('?')[0] + '?')
    answer_list.append(temp_a.split("?")[1].split('.')[0][:-1])
    question.append(temp_a.split("?")[1].split('.')[1] + '?')
    answer_list.append(temp_a.split("?")[2].split('.')[0][:-1])
    temp_a.split('?')[-1]
    temp_a.split("?")[0] 
    temp_a.split("?")[8]
    range(1,len(temp_a.split('?'))-1)
    temp_a.split('?')
    
    
    question = []
    answer_list = []
    answer = []
    
    # 첫번째 질문 저장 
    temp_a = df_a['text'].iloc[a]
    question.append(temp_a.split('?')[0] + '?')
    
    # 첫번째 문제부터 마지막 문제까지만 저장 
    for i in range(1, len(temp_a.split('?'))-1):
        answer_list.append(temp_a.split("?")[i].split('.')[0][:-1])
        question.append(temp_a.split("?")[i].split('.')[1] + '?')
    
    # 마지막 문제
    answer_list.append(temp_a.split('?')[-1].split('▶')[0])
    
    # 정답 처리
    a = temp_a.split('?')[-1].split('▶')[1]
    # a = a.replace(' ', '')  # 공백 제거
    a = a.replace(' ①', '①').replace(' ②', '②').replace(' ③', '③').replace(' ④', '④')
    answer = a.split(' ')[2:]

    df_a = pd.DataFrame()
    df_a['질문'] = question
    df_a['보기 답'] = answer_list
    df_a['답'] = answer
    
    dfs.append(df_a)  # 생성된 데이터프레임을 리스트에 추가


result = pd.concat(dfs)

result = result.reset_index(drop = True)
for i in result.index:
  result.loc[i, '답'] = result.loc[i, '답'][-1]


 
result.loc[result['답'] == '①', '답'] = 1
result.loc[result['답'] == '②', '답'] = 2

num = 1
for i in ['①', '②', '③', '④']:
  result.loc[result['답'] == i, '답'] = num
  num += 1
result['질문'].iloc[0] = result['질문'].iloc[0][2:]


# api key 불러오기
os.environ.get("jiji.api_key")
openai.api_key = os.environ["jiji.api_key"]


# 랜덤으로 질문 선택하기
random_question = result.sample(n=1)

# 선택한 질문 출력하기
qa = random_question['질문'].values[0]
print(random_question['질문'].values[0])

# 선택한 질문의 보기 답 출력하기
qb = random_question['보기 답'].values[0]
print(random_question['보기 답'].values[0])

# 답 정의해주기
qc = random_question['답'].values[0]

# 사용자로부터 답 입력받기
user_answer = input("질문에 대한 정답을 선택하세요: ")

# 정답과 사용자 입력 답 비교하기
while user_answer.strip() != str(random_question['답'].values[0]).strip():
    print("오답입니다. 다시 시도해주세요.")
    user_answer = input("질문에 대한 정답을 선택하세요: ")
else:
    print("정답입니다!")
    prompt = "질문 : {0}과  보기 : {1}, 그리고 답 : {2} 를 활용해 문제를 풀고 설명해주세요".format(qa,qb,qc)


    response = openai.Completion.create(
      model="text-davinci-003", # text-davinci-003라는 모델을 사용.
      prompt = prompt, # 질의응답
      temperature=1,# 0으로 되어있었음. 낮을수록 보수적, 높을수록 창의적
      max_tokens=1000, # 높을수록 말이 길게 나온다.
      top_p=1.0, # 다음에 나올 단어나 문장을 예측할 때 선택 가능한 단어나 문장 중 확률 값이 높은 것만 선택. 이 때 선택할 확률 값의 상위 몇 %를 선택할지를 지정
      frequency_penalty=0.0, # 생성하고자 하는 텍스트에서 자주 등장하는 단어나 문장의 확률 값을 낮춥니다. 이를 통해 생성 결과에 대한 다양성을 높일 수 있다.
      presence_penalty=0.0 # 생성하고자 하는 텍스트에서 이미 나온 단어나 문장의 확률 값을 낮춤. 이를 통해 생성 결과가 이전에 나온 단어나 문장과 중복되지 않도록 할 수 있다.
    )
    print(response['choices'][0]['text'].strip())



"나는 {0}색과 {1}색을 좋아해요".format("파란","빨간")


















