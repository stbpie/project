#!/usr/bin/env python
# coding: utf-8
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import streamlit as st
plt.rc('font', family='NanumGothic')
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from IPython.core.display import display, HTML
pyo.init_notebook_mode()

seoul_geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"

st.set_page_config(page_title="My Dashboard", page_icon=":bar_chart:", layout="wide")

with urlopen(seoul_geo_url) as response:
    seoul_geojson = json.load(response)
    
    

# %%
df_market = pd.read_csv('seoul_data/상권_추정매출.csv(2017_2021).csv')
df_lease = pd.read_csv('seoul_data/자치구_평당임대료.csv', encoding='cp949')
인구 = pd.read_csv("seoul_data/인구통합_최종.csv")
버스 = pd.read_csv("seoul_data/자치구_버스정류장위경도,개수_최종.csv")
지하철 = pd.read_csv("seoul_data/자치구별_지하철역위경도_최종.csv")
df_regression = pd.read_excel('seoul_data/regression.xlsx')


# %%

# 상권 매출정보와 평당 임대료 df를 넣으면 병합한 데이터 프레임을 준다
# 구별 평당 임대료, 월평균 매출, 객단가까지 함쳐서 준다
def get_dataframe(df_market, df_lease):
    temp_list = []
    
    df_market = df_market[df_market['기준_년_코드'] == 2021]
    df_lease = df_lease.set_index('Unnamed: 0')
    
#    구별 평당 임대료 데이터 넣기
    for gu in df_market['구'].unique():
        df_temp = df_market[df_market['구'] == gu].copy()
        df_temp['평당 임대료']= df_lease.loc[gu, '연평균임대료']
        temp_list.append(df_temp)
    df_temp = pd.concat(temp_list)
    
#     매장 월평균 매출 열 추가하기
    df_temp['매장 월평균 매출'] = df_temp['분기당_매출_금액'] / (df_temp['점포수'] * 3)
    
#     객단가 계산하기
    df_temp['객단가'] = df_temp['분기당_매출_금액'] / df_temp['분기당_매출_건수']
    
    return df_temp


# %%
df = get_dataframe(df_market, df_lease)


# %%
def get_sales_lease_top5(service_name, surface_area, df):
    
    df_temp = pd.DataFrame()
#     선택 업종의 데이터만 가져오기
    df_service = df[df['서비스_업종_코드_명'].str.contains(service_name)].copy()
    
#     구별 매출 평균, 평당 임대료 값 가져오기
    for gu in df['구'].unique():
        df_temp.loc[0, gu] = round(df_service[df_service['구'] == gu]['매장 월평균 매출'].mean(), 0)
        df_temp.loc[1, gu] = df_service[df_service['구'] == gu]['평당 임대료'].mean()
        df_temp.loc[2, gu] = round(df_service[df_service['구'] == gu]['객단가'].mean(), 0)
        df_temp.loc[3, gu] = df_service[df_service['구'] == gu]['lat'].mean()
        df_temp.loc[4, gu] = df_service[df_service['구'] == gu]['lot'].mean()
        
#   df_temp 형태 및 열 이름 바꿔주기      
    df_temp = df_temp.T
    df_temp.columns = ['월평균 매출', '평당 임대료', '평균 객단가', '위도', '경도']
    
#     월평균 매출 - 임대료 값, 임대료 비율 추가하기
    df_temp['매출-임대료'] = df_temp['월평균 매출'] - (df_temp['평당 임대료'] * surface_area)
    df_temp['임대료'] = df_temp['평당 임대료'] * surface_area
    df_temp['임대료 비율'] = (df_temp['평당 임대료'] * surface_area) / df_temp['월평균 매출'] * 100
    df_temp = df_temp.sort_values(by='평당 임대료', ascending = True) 
    
#     plt.figure(1)
#     plt.barh(df_temp['매출-임대료'].tail())
#     df_temp['매출-임대료'].tail().plot(kind='barh', 
#                                            title=f'{service_name} 업종 매출-임대료 상위 top5 구')
    return df_temp


# %%
def get_service_seoul_data(service_name, df):
    df_temp = pd.DataFrame()
    
    #     선택 업종의 데이터만 가져오기
    df_service = df[df['서비스_업종_코드_명'].str.contains(service_name)]
    
    #     해당 업종 2021년 전체 매출, 점포수 합
    df_temp.loc[0, '서울_전체_매출'] = df_service['분기당_매출_금액'].sum()
    df_temp.loc[0, '서울_전체_점포수'] = df_service[df_service['기준_분기_코드'] == 4]['점포수'].sum()
     
    #     2021년 분기별 매출합
    for no in range(1, 5):
        df_temp.loc[0, f'서울_전체_매출_{no}분기'] = df_service[df_service['기준_분기_코드'] == no]['분기당_매출_금액'].sum()
        
    #     주중, 주말 매출합
    df_temp.loc[0, '주중_매출합'] = df_service['주중_매출_금액'].sum()
    df_temp.loc[0, '주말_매출합'] = df_service['주말_매출_금액'].sum()
    
    #     남성, 여성 매출합
    df_temp.loc[0, '남성_매출합'] = df_service['남성_매출_금액'].sum()
    df_temp.loc[0, '여성_매출합'] = df_service['여성_매출_금액'].sum()
    df_temp.loc[0, '남성_객단가'] = df_service['남성_매출_금액'].sum() / df_service['남성_매출_건수'].sum()
    df_temp.loc[0, '여성_객단가'] = df_service['여성_매출_금액'].sum() / df_service['여성_매출_건수'].sum()

    #     요일별 매출 금액
    for day_name in list('월화수목금토일'):
        df_temp.loc[0, f'{day_name}_매출합'] = df_service[f'{day_name}요일_매출_금액'].sum()
    
    #     연령대별 매출 추이
    for no in range(1, 7):
        if no != 6:
            df_temp.loc[0, f'{no}0대_매출합'] = df_service[f'연령대_{no}0_매출_금액'].sum()
        else:
            df_temp.loc[0, f'{no}0대 이상_매출합'] = df_service[f'연령대_{no}0_이상_매출_금액'].sum()
    
    #     연령별 객단가
    for no in range(1, 7):
        if no != 6:
            df_temp.loc[0, f'{no}0대_객단가'] = df_service[f'연령대_{no}0_매출_금액'].sum() / df_service[f'연령대_{no}0_매출_건수'].sum()
        else:
            df_temp.loc[0, f'{no}0대 이상_객단가'] = df_service[f'연령대_{no}0_이상_매출_금액'].sum() / df_service[f'연령대_{no}0_이상_매출_건수'].sum()
    
     #     시간대 매출금액  '시간대_00~06_매출_금액'
    df_temp.loc[0, '00~06_매출합'] = df_service['시간대_00~06_매출_금액'].sum()
    df_temp.loc[0, '06~11_매출합'] = df_service['시간대_06~11_매출_금액'].sum()
    i = 11
    while True:
        
        if i != 17:
            df_temp.loc[0, f'{i}~{i+3}_매출합'] = df_service[f'시간대_{i}~{i+3}_매출_금액'].sum()
            i += 3
            if i >= 24:
                break
        else:
            df_temp.loc[0, f'{i}~{i+4}_매출합'] = df_service[f'시간대_{i}~{i+4}_매출_금액'].sum()
            i += 4
    
    
    return df_temp


# %% [markdown]
# ### 다중회귀분석

# %%
def regression_kind1(business_type, df_regression):
    
    # 특정 업종의 데이터프레임 생성
    merged_select = df_regression[df_regression['업종'].str.contains(business_type)].reset_index(drop=True)
    # 다중공산성 파악 - VIF
    y = merged_select['연간 총 매출금액']
    X = merged_select[['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']]
    X = sm.add_constant(X)
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["feature"] = X.columns
    
    
    # 스케일 조정
    scale_list = ['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']
    scale = StandardScaler()
    merged_select[scale_list] = scale.fit_transform(merged_select[scale_list])
    
    # 다중회귀분석
    X = merged_select[['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']]
    y = merged_select['연간 총 매출금액']
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit()
    reg_result = result.summary()
    
    
    aa = ""
    if result.rsquared_adj <= 0.7 or result.fvalue <= 4:
        aa = "주의: 모델의 설명력이 정확하지 않을 수 있습니다."
        # 독립변수의 유의확률 p값이 0.05보다 낮은 변수 출력
    significant_features = []
    bb = ""
    for i in range(len(result.pvalues)):
        if result.pvalues[i] < 0.05:
            significant_features.append(result.params.index[i])
            bb = f"종속변수와 통계적으로 유의한 관계를 가진 변수: {significant_features}"
    return [vif, reg_result,aa,bb]


# %% [markdown]
# ### Coefficients

# %%
def regression_kind2(business_type, df_regression):
    
    # 특정 업종의 데이터프레임 생성
    merged_select = df_regression[df_regression['업종'].str.contains(business_type)].reset_index(drop=True)
 
    
    # 스케일 조정
    scale_list = ['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']
    scale = StandardScaler()
    merged_select[scale_list] = scale.fit_transform(merged_select[scale_list])
    
    # 다중회귀분석
    X = merged_select[['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']]
    y = merged_select['연간 총 매출금액']
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit()
    reg_result = result.summary()
    
    
        
    # 회귀분석 결과에서 독립변수들의 계수 추출
    coef = result.params[1:]
    
    # 독립변수들의 계수와 유의확률을 데이터프레임으로 저장
    coef_df = pd.DataFrame({'coef': coef, 'pvalue': result.pvalues[1:]})
        
    # 유의확률이 0.05 이하인 변수들의 이름을 추출해서 리스트로 저장
    significant_vars = list(coef_df[coef_df['pvalue'] <= 0.05].index)
    
    # 독립변수들의 계수를 막대그래프로 시각화
    plt.rc('font', family = 'Malgun Gothic')
    plt.rc('axes', unicode_minus=False)  # '-' 기호 표시
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots()

    
    ax.bar(coef.index, coef.values, color='lightgreen')
    ax.bar(significant_vars, coef_df.loc[significant_vars, 'coef'], color='green')
    ax.set_title('Coefficients of Independent Variables')
    ax.set_xlabel('독립변수')
    ax.set_ylabel('계수')
    plt.xticks(rotation=45)
    plt.ticklabel_format(axis='y', style='plain')  # 지수표현 실수로 바꿈
#     for var in significant_vars:
#         print(f"'\033[1m\033[32m{var}\033[0m'는 한 단위 증가시 {coef[var]:.3f}만큼 '연간 총 매출금액'에 영향을 줍니다.")
    
 
    return fig


# %% [markdown]
# ### 독립변수

# %%
def regression_kind3(business_type, df_regression):
    
    # 특정 업종의 데이터프레임 생성
    merged_select = df_regression[df_regression['업종'].str.contains(business_type)].reset_index(drop=True)
    
    
    # 스케일 조정
    scale_list = ['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']
    scale = StandardScaler()
    merged_select[scale_list] = scale.fit_transform(merged_select[scale_list])
    
    # 다중회귀분석
    X = merged_select[['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']]
    y = merged_select['연간 총 매출금액']
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit()
        
    # 회귀분석 결과에서 독립변수들의 계수 추출
    coef = result.params[1:]
    
    # 독립변수들의 계수와 유의확률을 데이터프레임으로 저장
    coef_df = pd.DataFrame({'coef': coef, 'pvalue': result.pvalues[1:]})
        
    # 유의확률이 0.05 이하인 변수들의 이름을 추출해서 리스트로 저장
    significant_vars = list(coef_df[coef_df['pvalue'] <= 0.05].index)
    
        
    # 회귀분석 결과에서 계수와 표준오차 추출
    coef = result.params.values[1:]
    stderr = result.bse.values[1:]
    
    # 독립변수 이름 설정
    names = ['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']
    
    # 에러바 그리기

    plt.style.use('dark_background')
    palette = 'Blues_r'
    
    fig, ax = plt.subplots()
    ax.errorbar(names, coef, yerr=stderr, fmt='o', capsize=5)
    ax.set_xlabel('독립변수')
    ax.set_ylabel('계수')
    ax.set_title('각 독립변수의 계수와 표준오차')
    plt.xticks(rotation=45)
    plt.ticklabel_format(axis='y', style='plain')  # 지수표현 실수로 바꿈
    
    return fig


# %% [markdown]
# ### 버스

# %%
def plotbus(input_gu):
    
    seoul_geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
    with urlopen(seoul_geo_url) as response:
        seoul_geojson = json.load(response)

    gu_df = 버스[버스['자치구'].str.contains(input_gu)][['정류소명', '위도', '경도', '버스정류장 개수']]
    
    if len(gu_df) == 0:
        print("해당 자치구에는 버스정류장이 없습니다.")
    else:
        fig_bus = px.scatter_mapbox(gu_df,
                                lat="위도",
                                lon="경도",
                                color="버스정류장 개수",
                                hover_name="정류소명",
                                zoom=12,
                                center=dict(lat=gu_df['위도'].median(), lon=gu_df['경도'].median()),
                                mapbox_style='open-street-map', color_continuous_scale='Reds')
        fig_bus.update_traces(marker=dict(size=10)) 
        fig_bus.update_geos(fitbounds="locations", visible=False)
        fig_bus.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig_bus


# %% [markdown]
# ### 지하철

# %%
def plotsubway(input_gu):
    
    seoul_geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
    with urlopen(seoul_geo_url) as response:
        seoul_geojson = json.load(response)
        
    gu_df = 지하철[지하철['자치구'].str.contains(input_gu)][['역명','호선', '위도', '경도', '역개수']]
    
    if len(gu_df) == 0:
        print("해당 자치구에는 지하철이 없습니다.")
    else:
        fig_sub = px.scatter_mapbox(gu_df,
                                    lat="위도",
                                    lon="경도",
                                    color="호선",
                                    hover_name="역명",
                                    zoom=12,
                                    center=dict(lat=gu_df['위도'].median(), lon=gu_df['경도'].median()),
                                    mapbox_style='open-street-map', color_continuous_scale='Reds')
        fig_sub.update_traces(marker=dict(size=20)) 
        fig_sub.update_geos(fitbounds="locations", visible=False)
        fig_sub.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig_sub


# %% [markdown]
# ### 인구) (1) 연령, 세대별 인구 Pie 차트 

# %%
def in_gu1(gu):
    df_gu = 인구[인구['자치구'].str.contains(gu)]

    # 연령
    fig3 = go.Figure(data=[
           go.Pie(labels=['10대', '20대', '30대', '40대', '50대','60대 이상'],
               values=[df_gu['10대'].sum(), df_gu['20대'].sum(), df_gu['30대'].sum(),
                       df_gu['40대'].sum(), df_gu['50대'].sum(), df_gu['60대 이상'].sum()],
               textinfo='label+value+percent', hole=0.3, 
               marker=dict(colors=['#870808', '#a62e2e', '#f06060', '#f28080', '#e6a8a8', '#f7d5d5']))
    ])
    fig3.update_layout(title=f"{gu} 연령 분포", template='plotly_dark')
    
    # 세대별인구 
    fig4 = go.Figure(data=[
        go.Pie(labels = ['1인가구', '2인가구', '3인가구', '4인가구', '5인가구 이상'],
               values = [df_gu['1인세대'].sum(), df_gu['2인세대'].sum(), 
                         df_gu['3인세대'].sum(), df_gu['4인세대'].sum(), 
                         df_gu['5인세대 이상'].sum()],
               textinfo='percent+label+value', hole=0.3,
               marker=dict(colors=['#a62e2e', '#f06060', '#f28080', '#e6a8a8', '#f7d5d5'])
               )
    ])
    fig4.update_layout(title=f"{gu} 세대별인구 분포", template='plotly_dark')
    
    # subplot
    fig1 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig1.add_trace(fig3.data[0], row=1, col=1)
    fig1.add_trace(fig4.data[0], row=1, col=2)

    fig1.update_layout(title=f"{gu} 연령별, 세대별 인구 현황", template='plotly_dark')

    return fig1


# %% [markdown]
# ### 인구) (2) 주간인구, 학생인구 Pie 차트

# %%
def in_gu2(gu):
    df_gu = 인구[인구['자치구'].str.contains(gu)]

    
    # 주간인구 
    fig1 = go.Figure(data=[
        go.Pie(labels=['유입인구', '상주인구'], values=[df_gu['유입인구(소계)'].sum(), df_gu['상주인구(소계)'].sum()],
               textinfo='label+percent', hole=0.3,
              marker=dict(colors=['#a62e2e', '#f7d5d5']))
        
    ])
    fig1.update_layout(title=f"{gu} 주간인구 현황", template='plotly_dark')

     # 학생인구 
    fig3 = go.Figure(data=[
        go.Pie(labels = ['초등학생', '중학생', '고등학생', '대학생'],
               values = [df_gu['초등학교'].sum(), df_gu['중학교'].sum(), 
                         df_gu['고등학교'].sum(), df_gu['대학교'].sum()],
               textinfo='label+percent', hole=0.3,
              marker=dict(colors=['#a62e2e', '#f06060', '#f28080', '#e6a8a8']))
               
    ])
    fig3.update_layout(title=f"{gu} 학생인구 분포", template='plotly_dark')
    
     # subplot
    fig2 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig2.add_trace(fig1.data[0], row=1, col=1)
    fig2.add_trace(fig3.data[0], row=1, col=2)

    fig2.update_layout(title=f"{gu} 주간, 학생인구 현황", template='plotly_dark')

    return fig2


# %% [markdown]
# ### 인구) (3) 주간인구, 인구밀집도 Bar 차트

# %%
def in_gu3(gu):
    df_gu = 인구[인구['자치구'].str.contains(gu)]
    
    # 주간인구 bar
    x1 = ['주간인구']
    fig1 = go.Figure(data=[
        go.Bar(name='상주인구', x=x1, y=df_gu['상주인구(소계)'],
               text=df_gu['상주인구(소계)'], textposition='auto',
              marker=dict(color=['#e6a8a8'])),
        go.Bar(name='유입인구', x=x1, y=df_gu['유입인구(소계)'],
               text=df_gu['유입인구(소계)'], textposition='auto',
               marker=dict(color=['#a62e2e']))],
                    )

    # 인구밀집도 bar
    fig2 = go.Figure(data=[go.Bar(name='인구밀집도', x=['인구밀집도(10 000 m²당 인구수)'], y=df_gu['인구밀집도(10 000 m²당 인구수)'],
                           text=df_gu['인구밀집도(10 000 m²당 인구수)'], textposition='auto',
                           marker=dict(color=['#f28080']))])

    # 두개 세로로 출력
    fig3 = make_subplots(rows=2, cols=1)
    fig3.add_trace(fig2.data[0], row=2, col=1)
    fig3.add_trace(fig1.data[0], row=1, col=1)
    fig3.add_trace(fig1.data[1], row=1, col=1)
    fig3.update_yaxes(title_text='주간인구', row=1, col=1)
    fig3.update_yaxes(title_text='인구밀집도(10 000 m²당 인구수)', row=2, col=1)


    fig3.update_layout(height=600, title=f"{gu} 인구밀집도 및 주간인구 현황", template='plotly_dark')

    return fig3

# %%

# %%
with st.expander("==== 업종 참고(펼쳐보기) ===="):
 
    st.write(df['서비스_업종_코드_명'].unique())
    
service_name = st.text_input(label="업종을 입력해 주세요", value="커피")
surface_area = st.number_input(label="평수를 입력해 주세요", value=20)
service_search = st.button("Confirm")

gu_name = st.text_input(label="구 이름을 입력해 주세요")
gu_search = st.button("검색")

row1_1, row1_2= st.columns([1, 1])
row2_1, row2_2, row2_3 = st.columns([1, 1,1])
row3_1, row3_2 = st.columns([1, 1])
row4_1, row4_2, row4_3 = st.columns([1, 1, 1])
row5_1, row5_2 = st.columns([2, 1])

            
if service_search or gu_search:
    df_sales = get_sales_lease_top5(service_name, surface_area, df)
    df_several = get_service_seoul_data(service_name, df)
    df_sales['시군구'] = df_sales.index
    
    with row1_1:
        st.subheader(f'{service_name} 업종 매출 분석')
        fig = px.choropleth(df_sales, geojson=seoul_geojson, color="매출-임대료",
                        locations=df_sales.index, featureidkey="properties.name", labels="시군구명",
                        projection="mercator", color_continuous_scale='Blues')
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(title_text = f'{service_name} 업종 (매출-임대료) 비교_{surface_area}평 기준', 
                          title_font_size = 20,  width=800, height=600, template='plotly_dark')
    
        st.plotly_chart(fig)
    
    with row1_2:
        row1_2_1, row1_2_2= st.columns([1, 1])

        
        with row1_2_1:
            df_seoul_sales = df_several[['서울_전체_매출', '서울_전체_점포수']]
            df_gender = round(df_several[['남성_객단가', '여성_객단가']], -1)
            df_seoul_sales.columns = ['전체 매출합(원)', '점포수(개)']
            df_seoul_sales = df_seoul_sales.T
            df_seoul_sales.columns = ['내용']
            df_gender.index = ['객단가(원/인)']
            df_gender.columns = ['남성', '여성']
            
            st.dataframe(df_seoul_sales)
            st.dataframe(df_gender)
            
            df_gender_sales = df_several[['남성_매출합', '여성_매출합']]
            df_gender_sales.columns = ['남성', '여성']
            df_gender_sales = df_gender_sales.T
            
            fig = go.Figure(data=[
                go.Pie(labels=df_gender_sales.index,
                       values=df_gender_sales[0],
                       textinfo='label+value+percent', hole=0.3, 
                       marker=dict(colors=['#105CAC', '#A7CFE7']))
            ])
            fig.update_layout(title=f"성별 매출 분포", template='plotly_dark')
            st.plotly_chart(fig)
           
        with row1_2_2:
            st.write('hello')
               
    with row2_1:
        tab1, tab2= st.tabs(['수익 top5 구' , '수익 하위 top 5 구'])
        
        with tab1:
#    매출-임대료 top5 구
            df_plt1 = df_sales.sort_values(by='매출-임대료', ascending=False).head()
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (7, 4)
            plt.rc('font', family='NanumGothic')
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('dark_background')
            colors = sns.color_palette('Blues_r', len(df_plt1['시군구']))
            
            fig, ax1 = plt.subplots()

            ax1.bar(df_plt1['시군구'], df_plt1['매출-임대료'], color=colors, width=0.2, label='매출-임대료')
            ax1.axhline(df_sales['매출-임대료'].mean(),label='Mean', c='r', ls=':')

            ax1.set_xlabel('시군구')
            ax1.set_ylabel('매출-임대료 (원)')
            ax1.tick_params(axis='both', direction='in')

            ax2 = ax1.twinx()
            ax2.plot(df_plt1['시군구'], df_plt1['평균 객단가'], '-s', color='white', markersize=4, linewidth=2, alpha=0.7, label='객단가')
 
            ax2.set_ylabel('객단가 (원/인)')
            ax2.tick_params(axis='y', direction='in')

            ax1.set_zorder(ax2.get_zorder() - 10)
            ax1.patch.set_visible(False)

            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
            ax2.legend(loc='upper center', bbox_to_anchor=(0.85, -0.2))

            st.pyplot(fig)

        with tab2:
            df_plt2 = df_sales.sort_values(by='매출-임대료', ascending=False).tail()
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (7, 4)
            plt.rc('font', family='NanumGothic')
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('dark_background')
            plt.rcParams['font.size'] = 12
            colors = sns.color_palette('Blues_r', len(df_plt2['시군구']))
            
            fig, ax1 = plt.subplots()

            ax1.bar(df_plt2['시군구'], df_plt2['매출-임대료'], color=colors, width=0.2, label='매출-임대료')
            ax1.axhline(df_sales['매출-임대료'].mean(),label='Mean', c='r', ls=':')

            ax1.set_xlabel('시군구')
            ax1.set_ylabel('매출-임대료 (원)')
            ax1.tick_params(axis='both', direction='in')

            ax2 = ax1.twinx()
            ax2.plot(df_plt2['시군구'], df_plt2['평균 객단가'], '-s', color='white', markersize=4, linewidth=2, alpha=0.7, label='객단가')
         
            ax2.set_ylabel('객단가 (원/인)')
            ax2.tick_params(axis='y', direction='in')

            ax1.set_zorder(ax2.get_zorder() - 10)
            ax1.patch.set_visible(False)

            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
            ax2.legend(loc='upper center', bbox_to_anchor=(0.85, -0.2))

            st.pyplot(fig)
                
    with row2_2:
        tab1, tab2, tab3 = st.tabs(['분기별 매출', '요일별 매출' , '시간대별 매출'])
        
        with tab1:
            df_quarter = df_several[['서울_전체_매출_1분기', '서울_전체_매출_2분기', '서울_전체_매출_3분기', '서울_전체_매출_4분기']]
            df_quarter.columns = ['1분기', '2분기', '3분기','4분기']
            df_quarter = df_quarter.T
            df_quarter.columns = ['서울 전체 분기별 매출']
            
            plt.rcParams['figure.figsize'] = (7, 4)
            plt.style.use('dark_background')
            colors = sns.color_palette('Blues_r', len(df_plt1.index))
            fig, ax = plt.subplots() 
            
            ax.bar(df_quarter.index, df_quarter['서울 전체 분기별 매출'], color=colors,  width=0.2) 
            ax.plot(df_quarter.index, df_quarter['서울 전체 분기별 매출'], color='white', linestyle='--', marker='o')
                        
            st.pyplot(fig)
            
        with tab2:
            df_day = df_several[['월_매출합', '화_매출합', '수_매출합', '목_매출합', '금_매출합', '토_매출합', '일_매출합']]
            df_day.columns = list('월화수목금토일')
            df_day.index = ['매출']
            df_day = df_day.T
            
            plt.rcParams['figure.figsize'] = (7, 4)
            plt.style.use('dark_background')
            colors = sns.color_palette('Blues_r', len(df_day.index))
            
            fig, ax = plt.subplots()
            ax.bar(df_day.index, df_day['매출'], color=colors,  width=0.3) 
            ax.plot(df_day.index,  df_day['매출'], color='white', linestyle='--', marker='o') 

            st.pyplot(fig)
        
        with tab3:
            df_hour = df_several[['00~06_매출합', '06~11_매출합', '11~14_매출합', '14~17_매출합', '17~21_매출합', '21~24_매출합']]
            df_hour.columns = ['00~06', '06~11', '11~14', '14~17', '17~21', '21~24']
            df_hour.index = ['매출']
            df_hour = df_hour.T
            
            plt.rcParams['figure.figsize'] = (7, 4)
            plt.style.use('dark_background')
            colors = sns.color_palette('Blues_r', len(df_hour.index)) 
            
            fig, ax = plt.subplots()
            ax.bar(df_hour.index, df_hour['매출'], color=colors,  width=0.3) 
            ax.plot(df_hour.index,  df_hour['매출'], color='white', linestyle='--', marker='o') 

            st.pyplot(fig)
    
    with row2_3:
        
        st.write('연령대별 매출 추이')
        
        df_age = df_several[['10대_매출합', '20대_매출합', '30대_매출합', '40대_매출합', '50대_매출합', '60대 이상_매출합']]
        df_age_unit_price = df_several[['10대_객단가', '20대_객단가', '30대_객단가', '40대_객단가', '50대_객단가', '60대 이상_객단가']] 
        cols = ['10대', '20대', '30대', '40대', '50대', '60대 이상']
        df_age.columns =cols
        df_age_unit_price.columns = cols
        df_age.index = ['연령대별 매출']
        df_age_unit_price.index = ['연령대별 객단가']
        df_age = df_age.T
        df_age['연령대별 객단가'] = df_age_unit_price.T['연령대별 객단가']

        colors = sns.color_palette('Blues_r', len(df_age.index))
        plt.rcParams['figure.figsize'] = (7, 4)
        plt.rc('font', family='NanumGothic')
        plt.style.use('dark_background')
        plt.rcParams['font.size'] = 12
        fig, ax1 = plt.subplots()

        ax1.bar(df_age.index, df_age['연령대별 매출'], color=colors, width=0.2, label='매출')

        ax1.set_xlabel('연령대')
        ax1.set_ylabel('매출 (원)')
        ax1.tick_params(axis='both', direction='in')

        ax2 = ax1.twinx()
        ax2.plot(df_age.index, df_age['연령대별 객단가'], '-s', color='white', markersize=4, linewidth=2, alpha=0.7, label='객단가')
        ax2.set_ylabel('객단가 (원/인)')
        ax2.tick_params(axis='y', direction='in')

        ax1.set_zorder(ax2.get_zorder() - 10)
        ax1.patch.set_visible(False)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
        ax2.legend(loc='upper center', bbox_to_anchor=(0.85, -0.2))

       
        st.pyplot(fig)
        
# 업종 다중회귀분석 대시보드
    with row3_1:
        st.subheader(f'{service_name} 업종 다중회귀분석')
        st.sidebar.write(regression_kind1(service_name, df_regression))
        
    with row3_2:
        
        st.pyplot(regression_kind2(service_name, df_regression))
        
        st.pyplot(regression_kind3(service_name, df_regression))
        st.write('''
        에러바 그래프를 통해 각 독립변수의 종속변수에 대한 영향력의 정도와 통계적 유의성을 쉽게 파악할 수 있습니다.
        \033[1m\033[32m에러바의 위치\033[0m : 에러바는 독립변수 계수의 추정값을 중심으로 양쪽으로 그려집니다.이 때 계수의 추정값이 신뢰구간의 중심이 됩니다
        \033[1m\033[32m에러바의 길이\033[0m : 에러바의 길이는 각 독립변수 계수의 추정값의 표준오차를 나타냅니다. 길이가 짧을수록 해당 계수의 추정값이 더 정확하다는 것을 의미합니다.
        ''')    


# 자치구 대시보드
    if gu_search:
        with row4_1:
            st.subheader(f'\n\n{gu_name} 대시보드')
            st.plotly_chart(in_gu1(gu_name))
        with row4_2:
            st.subheader(' ')
            st.plotly_chart(in_gu2(gu_name))
        with row4_3:
            st.subheader(' ')
            tab1, tab2 = st.tabs(['버스 정류소 정보', '지하철역 정보'])
            with tab1:
                st.plotly_chart(plotbus(gu_name))
            with tab2:
                st.plotly_chart(plotsubway(gu_name))

        with row5_1:
            st.write('hi')

        with row5_2:
            st.plotly_chart(in_gu3(gu_name))



# %%

# %%
