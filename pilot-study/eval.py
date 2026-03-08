import pandas as pd

# 1. 다운받은 구글 시트 파일 이름을 여기에 씁니다.
file_name = 'mock_data.csv'

# 파이썬이 파일을 통째로 읽어옵니다! (직접 입력할 필요 X)
df = pd.read_csv(file_name)

# 2. 데이터를 예쁘게 세로로 담을 빈 공간
long_data = []

# 3. 파이썬이 한 줄씩(한 명씩) 읽으면서 10판 데이터를 세로로 알아서 자릅니다.
for index, row in df.iterrows():
    pid = row['참가자 ID'] 
    
    # 10판 반복 (구글 폼 구조상 4번째 열(인덱스 3)부터 5칸씩 반복됨)
    for i in range(10):
        col_idx = 3 + (i * 5) 
        
        # 만약 빈 칸이면 건너뛰는 안전장치
        if pd.isna(row.iloc[col_idx]):
            continue
            
        long_data.append({
            '참가자ID': pid,
            '라운드': i + 1,
            '조건(실험번호)': row.iloc[col_idx],      
            '도움됨': row.iloc[col_idx + 1],
            '예측가능': row.iloc[col_idx + 2],
            '조화로움': row.iloc[col_idx + 3],
            '즐거움': row.iloc[col_idx + 4]
        })

# 4. 세로형 데이터로 변환
df_long = pd.DataFrame(long_data)

# 5. 파이썬이 알아서 조건별(C-8, C-9 등) 평균을 쫙 계산합니다.
summary = df_long.groupby('조건(실험번호)')[['도움됨', '예측가능', '조화로움', '즐거움']].mean()

# 6. 계산된 결과를 새로운 엑셀(CSV) 파일로 저장합니다!
summary.to_csv('최종_분석결과.csv', encoding='utf-8-sig')

print("✅ 분석이 완료되어 '최종_분석결과.csv' 파일이 생성되었습니다!")

# 7. 소수점 둘째 자리까지만 깔끔하게 출력
summary = summary.round(2)
print("📊 조건별 평균 분석 결과:")
print(summary)

# 8. 일단 기본 형태의 CSV로 저장합니다.
file_out = '최종_분석결과_한칸띄움.csv'
summary.to_csv(file_out, encoding='utf-8-sig')

# 9. 엑셀 파일에 한 줄씩 빈칸(엔터 두 번)을 넣기 위해 덮어쓰는 작업
with open(file_out, 'r', encoding='utf-8-sig') as f:
    content = f.read()

# 기존 엔터(\n)를 엔터 두 번(\n\n)으로 바꿔서 다시 저장!
with open(file_out, 'w', encoding='utf-8-sig') as f:
    f.write(content.replace('\n', '\n\n'))

print(f"\n✅ 완료! 엑셀을 열어보시면 '{file_out}' 파일이 한 줄씩 예쁘게 띄워져 있을 겁니다!")
