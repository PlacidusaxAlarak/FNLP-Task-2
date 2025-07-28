import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def plot_all_results(df):
    sns.set_style('whitegrid')#设置绘图风格
    plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False#正常显示负号

    groups=df['group'].unique()

    #为每个组创建一张图
    for group in groups:
        plt.figure(figsize=(10, 6))

        #筛选出当前组的数据
        group_df=df[df['group']==group].sort_values(by='accuracy', ascending=False)#降序

        #创建条形图
        barplot=sns.barplot(x='accuracy', y='param_value', data=group_df, palette='viridis', orient='h')

        #在条形图上显示数值
        for index, row in group_df.iterrows():
            plt.text(row.accuracy, index, f'{row.accuracy:.2f}%', color='black', ha='left', va='center')

            plt.title(f'{group}对模型性能的影响', fontsize=16)
            plt.xlabel('最佳验证集准确率(%)', fontsize=12)
            plt.ylabel('参数值', fontsize=12)
            plt.xlim(0, max(group_df['accuracy'])*1.15)

            #保存图标
            save_dir='./results/charts'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename=f"{group.replace(' ', '_')}_comparison.png"
            save_pth=os.path.join(save_dir, filename)

            plt.tight_layout()
            plt.savefig(save_pth)
            print(f"Chart saved to {save_pth}")

        plt.show()