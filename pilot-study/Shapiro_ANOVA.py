import pandas as pd
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore') # 보기 싫은 경고 메시지 숨김

# 1. 파일 불러오기
file_name = 'mock_data.csv'
df = pd.read_csv(file_name)

# 2. 데이터를 세로형(Long format)으로 변환
long_data = []
for index, row in df.iterrows():
    pid = row['참가자 ID'] 
    for i in range(10):
        col_idx = 3 + (i * 5) 
        if pd.isna(row.iloc[col_idx]):
            continue
        long_data.append({
            '참가자ID': pid,
            'Condition': row.iloc[col_idx],      
            '도움됨': row.iloc[col_idx + 1],
            '예측가능': row.iloc[col_idx + 2],
            '조화로움': row.iloc[col_idx + 3],
            '즐거움': row.iloc[col_idx + 4]
        })

df_long = pd.DataFrame(long_data)

# 4개 문항 모두 확실하게 숫자로 변환
target_vars = ['도움됨', '예측가능', '조화로움', '즐거움']
for var in target_vars:
    df_long[var] = pd.to_numeric(df_long[var], errors='coerce')


# =========================================================
# 📊 4개 항목 반복 통계 분석 시작
# =========================================================

for var in target_vars:
    print("\n" + "="*60)
    print(f" 🎯 분석 항목: [{var}]")
    print("="*60)
    
    # [1] 정규성 검정 (Shapiro-Wilk Test)
    not_normal_count = 0
    for condition, group in df_long.groupby('Condition'):
        data_points = group[var].dropna()
        if len(data_points) >= 3:
            stat, p_value = stats.shapiro(data_points)
            if p_value < 0.05:
                not_normal_count += 1
    print(f"✔️ 정규성 검정 위배(p < 0.05) 조건: 총 30개 중 {not_normal_count}개")
    
    # [2] 프리드먼 검정 (Friedman Test - 비모수)
    try:
        groups = [group[var].dropna().values for name, group in df_long.groupby('Condition')]
        stat, p = stats.friedmanchisquare(*groups)
        print(f"✔️ 프리드먼 검정: Chi-square = {stat:.4f}, p-value = {p:.4e}")
        if p < 0.05:
            print("   👉 [통계적 유의미함] 에이전트 조건에 따라 뚜렷한 차이가 존재합니다!")
        else:
            print("   👉 [무의미함] 에이전트 조건 간 차이가 발견되지 않았습니다.")
    except Exception as e:
        print("프리드먼 검정 에러:", e)

    # [3] 반복측정 분산분석 (RM-ANOVA - 모수)
    try:
        rm_anova = AnovaRM(data=df_long, depvar=var, subject='참가자ID', within=['Condition'])
        res = rm_anova.fit()
        p_val_anova = res.anova_table['Pr > F'][0]
        f_val = res.anova_table['F Value'][0]
        print(f"✔️ RM-ANOVA 검정: F Value = {f_val:.4f}, p-value = {p_val_anova:.4e}")
    except Exception as e:
        print("RM-ANOVA 에러:", e)

print("\n✅ 모든 항목의 통계 분석이 완료되었습니다!")


import itertools
from statsmodels.stats.multitest import multipletests

print("\n" + "="*60)
print(" 🏆 1:1 사후 분석 (Post-hoc Analysis: Wilcoxon & Bonferroni)")
print("="*60)

# 💡 연구자님이 '직접 비교해보고 싶은' 핵심 조건들만 리스트에 넣으세요!
# 예시: C맵에서 '아무 정보 안 줌(C-0)' vs '행동+하이라이트(C-8)' vs '행동+텍스트(C-9)'
conditions_to_compare = ['C-0', 'C-8', 'C-9']

# 분석할 설문 항목을 선택합니다 (도움됨, 예측가능, 조화로움, 즐거움 중 택 1)
target_var = '도움됨'

print(f"✔️ 비교할 조건들: {conditions_to_compare}")
print(f"✔️ 분석할 항목: [{target_var}]\n")

# 선택한 조건들끼리 1:1로 짝을 짓습니다 (예: C-0 vs C-8, C-0 vs C-9, C-8 vs C-9)
comparisons = list(itertools.combinations(conditions_to_compare, 2))
p_values = []
valid_comparisons = []

for cond1, cond2 in comparisons:
    # 참가자 ID 순서대로 정렬하여 두 그룹의 데이터를 1:1로 매칭합니다
    data1 = df_long[df_long['Condition'] == cond1].sort_values('참가자ID')[target_var].dropna()
    data2 = df_long[df_long['Condition'] == cond2].sort_values('참가자ID')[target_var].dropna()
    
    # 두 데이터의 길이가 같고 최소 3개 이상일 때만 분석 실행
    if len(data1) == len(data2) and len(data1) >= 3:
        # 비모수 쌍체 검정 (Wilcoxon signed-rank test)
        stat, p = stats.wilcoxon(data1, data2)
        p_values.append(p)
        valid_comparisons.append((cond1, cond2))
    else:
        print(f"⚠️ {cond1} vs {cond2}: 결측치가 있거나 데이터 개수가 맞지 않아 제외됩니다.")

# 본페로니 교정 (여러 번 비교할수록 기준을 엄격하게 만들어서 가짜 결과를 걸러냄)
if p_values:
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

    print(f"📊 1:1 비교 결과 (본페로니 교정 적용됨):")
    for i, (cond1, cond2) in enumerate(valid_comparisons):
        is_significant = "🌟 유의미함" if reject[i] else "무의미함"
        print(f" - {cond1} vs {cond2} | 교정된 p-value: {pvals_corrected[i]:.4f} ({is_significant})")
else:
    print("비교할 수 있는 유효한 데이터가 없습니다.")