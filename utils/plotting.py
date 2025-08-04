import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_all_results(df):
    sns.set_style('whitegrid')
    # 确保中文字体文件存在，或者换成系统里有的，比如 'Microsoft YaHei' for Windows
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    groups = df['group'].unique()

    # 为每个组创建一张图
    for group in groups:
        # 1. 为每个新组创建一个新的 Figure
        plt.figure(figsize=(10, 6))

        # 2. 筛选出当前组的数据，并按准确率降序排序
        #    这是为了让之后的条形图按从上到下性能递减的方式显示
        group_df = df[df['group'] == group].sort_values(by='accuracy', ascending=False)

        # 3. 创建条形图，Seaborn会按照传入DataFrame的顺序来绘制y轴
        barplot = sns.barplot(x='accuracy', y='param_value', data=group_df, palette='viridis', orient='h')

        # 4. 在条形图上显示数值 (将这个逻辑移出循环)
        #    直接遍历已经画好的条形 (patches) 是更稳妥的方式
        for i, bar in enumerate(barplot.patches):
            # 获取条形的宽度 (即 accuracy 值)
            width = bar.get_width()
            # 在条形末尾的右侧一点点显示文本
            plt.text(width + 0.01 * plt.xlim()[1],  # x坐标：在条形末端再往右一点
                     bar.get_y() + bar.get_height() / 2, # y坐标：条形的垂直中心
                     f'{width:.2f}%', # 显示的文本
                     color='black', 
                     ha='left',       # 水平对齐：左对齐
                     va='center')     # 垂直对齐：居中

        # 5. 将所有绘图设置命令放在循环的外部，每个图只设置一次
        plt.title(f'{group} 对模型性能的影响', fontsize=16)
        plt.xlabel('最佳验证集准确率(%)', fontsize=12)
        plt.ylabel('参数值' if group != "Model Architecture" else "模型架构", fontsize=12) # Y轴标签可以更智能
        
        # 动态设置x轴范围，留出空间给文本
        plt.xlim(0, max(group_df['accuracy']) * 1.15) 

        # 6. 保存图表 (每个组只保存一次)
        save_dir = './results/charts'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 清理文件名，防止特殊字符导致问题
        safe_group_name = "".join([c for c in group if c.isalpha() or c.isdigit() or c in (' ', '_')]).rstrip()
        filename = f"{safe_group_name.replace(' ', '_')}_comparison.png"
        save_path = os.path.join(save_dir, filename)

        plt.tight_layout() # 调整布局防止标签重叠
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")

        # 7. 显示当前图表，并关闭它以释放内存，避免影响下一个图
        plt.show()
        plt.close()