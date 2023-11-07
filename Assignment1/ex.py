# nested_list = [[1, 2, 3], [2, 5], [2, 7]]
#
# # 使用集合来检测重复子列表
# seen = set()
# duplicates = []
#
# for sublist in range(len(nested_list)):
#     for i in nested_list[sublist]:
#         for lists in nested_list[sublist+1:]:
#             if nested_list[sublist] != lists:
#                 for j in lists:
#                     if i == j:
#                         duplicates.append(j)
#
#
# # 打印重复子列表
# # for duplicate in duplicates:
# #     print(duplicate)
# print(duplicates)

# a = [1,2,3,4]
# b = []
# b.append(a)
# b.append(a)
# print(len(b))

my_list = [
    [100, 143, 176, 209, 406, 482, 572, 573, 627, 675, 724, 728, 793, 794, 1024, 1043, 1046, 1160, 1162, 1185, 1194, 1214, 1247, 1305, 1324, 1390, 1393, 1456, 1457, 1464, 1469, 1470, 1485, 1531, 1590, 1614, 1616, 1671, 1678, 1826, 1890, 1927, 1935, 2059, 2113, 2125, 2128, 2156, 2157, 2160, 2162, 2177, 2210, 2262, 2310, 2316, 2323, 2359, 2361, 2367, 2381, 2484, 2500, 2516, 2538, 2561, 2570, 2594, 2607, 2626, 2648, 2725, 2756, 2877, 2885, 2927, 2936, 2954, 3030, 3032, 3074, 3095, 3103, 3164],
    [236, 397, 598, 1046, 1155, 1272, 1278, 1333, 1665, 1938, 2051, 2229, 2304, 2572, 2628, 2688, 2874, 3076, 3181],
    [],
    [],
    [],
    [1717],
    []
]
my_list = [[1, 2, 3], [2, 5], [3, 2, 4]]

# 创建一个字典来记录每个数字出现的次数
count_dict = {}

# 遍历所有子列表，统计数字出现次数
for sublist in my_list:
    for number in sublist:
        count_dict[number] = count_dict.get(number, 0) + 1

# 根据数字出现次数进行排序
sorted_numbers = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
sort = []
# 输出排序后的数字和它们的出现次数
for number, count in sorted_numbers[:10]:
    # print(f"数字 {number} 出现了 {count} 次")
    sort.append(number)
print(sort)

