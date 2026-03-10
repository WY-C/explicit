import pandas as pd
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
import warnings
import itertools
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore') # 보기 싫은 경고 메시지 숨김

# 1. 파일 불러오기
file_name = 'data.csv'
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
    
    # 분석에 사용할 수 있도록 결측치 제거
    df_clean = df_long.dropna(subset=[var])
    
    # [1] 정규성 검정 (Shapiro-Wilk Test)
    not_normal_count = 0
    condition_groups = df_clean.groupby('Condition')
    total_conditions = len(condition_groups)
    
    for condition, group in condition_groups:
        data_points = group[var]
        if len(data_points) >= 3:
            stat, p_value = stats.shapiro(data_points)
            if p_value < 0.05:
                not_normal_count += 1
    print(f"✔️ 정규성 검정 위배(p < 0.05) 조건: 총 {total_conditions}개 중 {not_normal_count}개")
    
    # 통계 분석을 위해 데이터를 참가자별-조건별 피벗 테이블로 변환 (결측치 확인용)
    pivot_df = df_clean.pivot(index='참가자ID', columns='Condition', values=var)
    
    # [2] 프리드먼 검정 (Friedman Test - 비모수)
    # 프리드먼 검정은 모든 조건을 수행한 참가자의 데이터만 필요하므로 결측치가 없는 행만 추출
    balanced_df = pivot_df.dropna()
    if len(balanced_df) >= 3 and balanced_df.shape[1] > 1:
        try:
            # balanced_df에서 각 열(Condition) 데이터를 추출하여 리스트로 묶음
            groups = [balanced_df[col].values for col in balanced_df.columns]
            stat, p = stats.friedmanchisquare(*groups)
            print(f"✔️ 프리드먼 검정 (완전한 데이터 {len(balanced_df)}명 기준): Chi-square = {stat:.4f}, p-value = {p:.4e}")
            if p < 0.05:
                print("   👉 [통계적 유의미함] 에이전트 조건에 따라 뚜렷한 차이가 존재합니다!")
            else:
                print("   👉 [무의미함] 에이전트 조건 간 차이가 발견되지 않았습니다.")
        except Exception as e:
            print(f"프리드먼 검정 에러: {e}")
    else:
        print("✔️ 프리드먼 검정: 모든 조건을 수행한 참가자가 부족하여(3명 미만) 실행할 수 없습니다.")

    # [3] 반복측정 분산분석 (RM-ANOVA - 모수)
    # AnovaRM 역시 균형 잡힌 데이터(결측치 없음)가 필수
    if len(balanced_df) >= 3 and balanced_df.shape[1] > 1:
        try:
            # 분석 가능한 완전한 참가자 ID만 필터링하여 다시 Long format 형태로 전달
            valid_pids = balanced_df.index
            df_anova = df_clean[df_clean['참가자ID'].isin(valid_pids)]
            
            rm_anova = AnovaRM(data=df_anova, depvar=var, subject='참가자ID', within=['Condition'])
            res = rm_anova.fit()
            p_val_anova = res.anova_table['Pr > F'][0]
            f_val = res.anova_table['F Value'][0]
            print(f"✔️ RM-ANOVA 검정 (완전한 데이터 {len(balanced_df)}명 기준): F Value = {f_val:.4f}, p-value = {p_val_anova:.4e}")
        except Exception as e:
            print(f"RM-ANOVA 에러: {e}")
    else:
        print("✔️ RM-ANOVA 검정: 모든 조건을 수행한 참가자가 부족하여(3명 미만) 실행할 수 없습니다.")

print("\n✅ 모든 항목의 통계 분석이 완료되었습니다!")


print("\n" + "="*60)
print(" 🏆 1:1 사후 분석 (Post-hoc Analysis: Wilcoxon & Bonferroni)")
print("="*60)

# 💡 연구자님이 '직접 비교해보고 싶은' 핵심 조건들만 리스트에 넣으세요!
conditions_to_compare = ['A-0', 'A-1', 'A-9'] # 제공된 데이터에 존재하는 조건으로 예시 변경

# 분석할 설문 항목을 선택합니다
target_var = '도움됨'

print(f"✔️ 비교할 조건들: {conditions_to_compare}")
print(f"✔️ 분석할 항목: [{target_var}]\n")

comparisons = list(itertools.combinations(conditions_to_compare, 2))
p_values = []
valid_comparisons = []

for cond1, cond2 in comparisons:
    # 단순 정렬이 아닌, '참가자ID'를 기준으로 두 조건 데이터를 정확히 병합 (Inner Join)
    df1 = df_long[df_long['Condition'] == cond1][['참가자ID', target_var]].rename(columns={target_var: 'val1'})
    df2 = df_long[df_long['Condition'] == cond2][['참가자ID', target_var]].rename(columns={target_var: 'val2'})
    
    merged = pd.merge(df1, df2, on='참가자ID', how='inner').dropna()
    
    if len(merged) >= 3:
        stat, p = stats.wilcoxon(merged['val1'], merged['val2'])
        p_values.append(p)
        valid_comparisons.append((cond1, cond2, len(merged)))
    else:
        print(f"⚠️ {cond1} vs {cond2}: 두 조건을 모두 수행한 참가자가 부족하여 제외됩니다. (현재 {len(merged)}명)")

# 본페로니 교정
if p_values:
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

    print(f"\n📊 1:1 비교 결과 (본페로니 교정 적용됨):")
    for i, (cond1, cond2, count) in enumerate(valid_comparisons):
        is_significant = "🌟 유의미함" if reject[i] else "무의미함"
        print(f" - {cond1} vs {cond2} (비교 인원: {count}명) | 교정된 p-value: {pvals_corrected[i]:.4f} ({is_significant})")
else:
    print("\n비교할 수 있는 유효한 데이터가 없습니다.")