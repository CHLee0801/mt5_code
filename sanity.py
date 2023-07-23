import pandas as pd

a = pd.read_csv("result/out.csv")
cnt = 0
correct = 0
for idx, row in a.iterrows():
    a = row['정답(도로명주소)']
    b = row['모델아웃풋']
    cnt += 1

    a = a.split('(')[0].strip()
    b = b.split('(')[0].strip()
    a = a.replace('지하','')
    b = b.replace('지하','')
    if a == b:
        correct += 1


print(correct)
print(cnt)
print(correct/cnt)