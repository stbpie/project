# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:01:36 2023

@author: NADA
"""

import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from IPython.core.display import display, HTML

df = pd.read_excel('regression.xlsx')

def regression_kind(business_type):
    
    # 특정 업종의 데이터프레임 생성
    merged_select = df[df['업종'].str.contains(business_type)].reset_index(drop=True)
    # 다중공산성 파악 - VIF
    y = merged_select['연간 총 매출금액']
    X = merged_select[['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']]
    X = sm.add_constant(X)
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["feature"] = X.columns
    print(vif)
    
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
    print(result.summary())
    
    # 주의 메시지 출력
    if result.rsquared_adj <= 0.7 or result.fvalue <= 4:
        print("\033[1m\033[31m주의: 모델의 설명력이 정확하지 않을 수 있습니다.\033[0m")
    
    # 독립변수의 유의확률 p값이 0.05보다 낮은 변수 출력
    significant_features = []
    for i in range(len(result.pvalues)):
        if result.pvalues[i] < 0.05:
            significant_features.append(result.params.index[i])
    print(f"\033[1m\033[32m종속변수와 통계적으로 유의한 관계를 가진 변수\033[0m: {significant_features}")
    
    
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
    palette = 'Blues_r'
    plt.xticks(rotation=45)
    plt.bar(coef.index, coef.values, color='lightgreen')
    plt.bar(significant_vars, coef_df.loc[significant_vars, 'coef'], color='green')
    plt.title('Coefficients of Independent Variables')
    plt.xlabel('독립변수')
    plt.ylabel('계수')
    plt.ticklabel_format(axis='y', style='plain')  # 지수표현 실수로 바꿈
    plt.show()
    
    print("\n유의확률 0.05 미만 독립변수들:")
    for var in significant_vars:
        print(f"'\033[1m\033[32m{var}\033[0m'는 한 단위 증가시 {coef[var]:.3f}만큼 '연간 총 매출금액'에 영향을 줍니다.")
    
    
    # 회귀분석 결과에서 계수와 표준오차 추출
    coef = result.params.values[1:]
    stderr = result.bse.values[1:]
    
    # 독립변수 이름 설정
    names = ['평당임대료', '평균소득', '학생비율', 'ha당 유동인구', '지하철역수', '버스정류장 개수', '주간인구(소계)']
    
    # 에러바 그리기
    plt.rc('font', family = 'Malgun Gothic')
    plt.style.use('dark_background')
    palette = 'Blues_r'
    fig, ax = plt.subplots()
    ax.errorbar(names, coef, yerr=stderr, fmt='o', capsize=5)
    ax.set_xlabel('독립변수')
    ax.set_ylabel('계수')
    ax.set_title('각 독립변수의 계수와 표준오차')
    plt.xticks(rotation=45)
    plt.ticklabel_format(axis='y', style='plain')  # 지수표현 실수로 바꿈
    plt.show()
    
    print("에러바 그래프를 통해 각 독립변수의 종속변수에 대한 영향력의 정도와 통계적 유의성을 쉽게 파악할 수 있습니다.")
    print("\033[1m\033[32m에러바의 위치\033[0m : 에러바는 독립변수 계수의 추정값을 중심으로 양쪽으로 그려집니다.이 때 계수의 추정값이 신뢰구간의 중심이 됩니다")
    print("\033[1m\033[32m에러바의 길이\033[0m : 에러바의 길이는 각 독립변수 계수의 추정값의 표준오차를 나타냅니다. 길이가 짧을수록 해당 계수의 추정값이 더 정확하다는 것을 의미합니다.")
    
regression_kind('네일')
