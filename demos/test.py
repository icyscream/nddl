import re
import pandas as pd


if __name__ == '__main__':

    # 读取文本文件
    with open('C:/Users/26411/Desktop/usage.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    # 使用正则表达式提取CPU使用率和显存占用率
    cpu_pattern = re.compile(r'CPU使用率： (\d+\.\d+)%')
    mem_pattern = re.compile(r'显存占用率： (\d+)MiB')
    cpu_usage = cpu_pattern.findall(data)
    mem_usage = mem_pattern.findall(data)

    # 将数据存入dataframe并写入Excel文件
    df = pd.DataFrame.from_dict({'CPU使用率': cpu_usage, '显存占用率': mem_usage}, orient='index')
    df.to_excel('C:/Users/26411/Desktop/output.xlsx', index=False)
