import ast

# path1 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\RRTpath.txt'
# path2 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\RRTpath_2023.txt'
# path3 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\RRTpath_2024.txt'
# path4 = r'E:\files\code\硕士论文code\Chaper2\finalresult\3d_result\env3\txt\RRTpath_2025.txt'
# path_load = [path1,path2,path3,path4]
#
#
# def read_mult_txt(path_load):
#     loaded_path = []
#     for path in path_load:
#         with open(path, 'r') as f:
#             loaded_path.append([ast.literal_eval(line.strip()) for line in f if line.strip()])  # 改为列表解析
#     return loaded_path
# load_path = read_mult_txt(path_load)

file = r'E:\files\code\硕士论文code\Chaper2\Dagger.txt'

with open(file, "r", encoding="utf-8") as file:
    content = file.read()

content = content.replace("[", "(").replace("]", ")")

with open(r'E:\files\code\硕士论文code\Chaper2\Dagger.txt', "w", encoding="utf-8") as file:
    file.write(content)



#
# with open(path_load, 'r') as f:
#     loaded_path = [ast.literal_eval(line.strip()) for line in f if line.strip()]  # 安全解析






