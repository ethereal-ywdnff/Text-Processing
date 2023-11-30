import pandas as pd

file1 = pd.read_csv('SampleSubmission_test_predictions_3classes_acpXXjd.tsv', delimiter='\t')
file2 = pd.read_csv('test_predictions_3classes_ace21kl.tsv', delimiter='\t')

# 比较两个DataFrame
# if file1.equals(file2):
#     print("1")
# else:
#     print("2")

# 初始化计数器
count = 0

for i in range(len(file2)):
    if file1.loc[i, 'Sentiment'] == file2.loc[i, 'Sentiment']:
        count += 1

# 输出匹配的句子数量
print(f"匹配的数量: {count}")