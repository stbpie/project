# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 10:25:51 2023

@author: NADA
"""

'''스타벅스 크롤링'''
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import requests
import pandas as pd

# 홈페이지 들어가기
driver = webdriver.Chrome()
driver.get('https://www.starbucks.co.kr/store/store_map.do?disp=locale')
time.sleep(3)
# 팝업창 끄기
'''try :
    driver.find_element_by_xpath('//*[@id="todayPop"]').click()
except :
    print("알림창이 없습니다")
    time.sleep(3)
    
pop = driver.find_element(By.CLASS_NAME, 'todayPop_txt christmas_pop_txt')
pop.click()'''

#pop = driver.find_element(By.XPATH, '/html/body/div[4]/p/a') #팝업창 끄기
#pop.click()
#time.sleep(3)
# 서울 버튼 누르기
seoul = driver.find_element(By.CLASS_NAME, 'set_sido_cd_btn')
seoul.click()
time.sleep(3)
# 전체 버튼 누르기
entire = driver.find_element(By.CLASS_NAME, 'set_gugun_cd_btn')
entire.click()
time.sleep(3)
# 매장정보
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
SBlist = soup.select('li.quickResultLstCon')
len(SBlist)

SBstore = SBlist[0]
SBstore['data-name'] #지점명
SBstore.find("p", attrs={"class":"result_details"}).text #주소
SBstore['data-lat'] # 위도
SBstore['data-long'] # 경도
SBstore.select('i')[0]['class'][0][4:] #타입

store_list = []
for item in SBlist:
    name = item['data-name']
    lat = item['data-lat']
    long = item['data-long']
    address = item.find("p", attrs={"class":"result_details"}).text
    storetype = item.select('i')[0]['class'][0][4:]
    store_list.append([name, lat, long, address, storetype])

len(store_list)

table = pd.DataFrame(store_list, columns = ['매장명','위도','경도','주소','매장타입'])
table.head()