import os
import pandas as pd
import time

from configs import Config
from train import run_training
from utils.plotting import plot_all_results

def main():

    results_data=[]
    base_config=Config()

    #实验1：不同学习率对性能的影响
    print("\n"+"="*50)
    print("Running Experiment1:Different Learning Rates")
    print("="*50)
    learning_rates=[1e-3, 5e-3, 1e-4, 1e-5]
    for lr in learning_rates:
        print(f"\n------Testing Learning Rate:{lr}")
        config = Config()
        config.update_params(
            model_name='TextCNN',
            learning_rate=lr,
            num_epochs=15
        )
        best_acc, duration = run_training(config)
        results_data.append({
            'group': "Learning Rate",
            'param_value': str(lr),
            'accuracy': best_acc,
            'duration_sec': duration
        })

    #实验2：优化器的影响
    print("\n"+"="*50)
    print("Running Experiment 2:Different Optimizers")
    print("="*50)
    optimizers=['Adam', 'SGD']
    for opt in optimizers:
        print(f"\n------Test Optimizer:{opt}-------")
        config=Config()
        #SGD需要更好的学习率
        lr_for_opt=1e-4 if opt=='Adam' else 1e-2
        config.update_params(
            model_name='TextCNN',
            optimizer=opt,
            learning_rate=lr_for_opt,
            num_epochs=15
        )
        best_acc, duration=run_training(config)
        results_data.append({
            'group':"Optimizer",
            'param_value':opt,
            'accuracy':best_acc,
            'duration_sec':duration
        })

    #实验3：不用卷积核配置的影响
    print("\n"+"="*50)
    print("Running Experiment 3:Different CNN Kernels")
    print("="*50)
    kernel_configs=[
        {'num_filters':128, 'filter_sizes':[2, 3, 4]},#多少个卷积核,之后会拼接成128*3=384维度的向量
        {'num_filters':128, 'filter_sizes':[3, 4, 5]},
        {'num_filters':256, 'filter_sizes':[3, 4, 5,]},
        {'num_filters':128, 'filter_sizes':[2, 3, 4, 5]}
    ]
    for k_config in kernel_configs:
         param_str=(f"Filters:{k_config['num_filters']}, Sizes:{k_config['filter_sizes']}")
         print(f"\n------Testing Kernel Config:{param_str}-------")
         config=Config()
         config.update_params(
             model_name='TextCNN',
             num_filters=k_config['num_filters'],
             filter_sizes=k_config['filter_sizes'],
             num_epochs=15
         )
         best_acc, duration=run_training(config)
         results_data.append({
             'group':"Kernel Config",
             'param_value':param_str,
             'accuracy':best_acc,
             'duration_sec':duration
         })

    #实验4：使用/不使用Glove预训练向量
    print("\n"+'='*50)
    print("Running Experiment 4:Glove vs Random Initalization")
    print('='*50)
    use_glove_options=[True, False]
    for use_glove in use_glove_options:
        param_str='With Glove' if use_glove else "Random Init"
        print(f"\n------Testing Embedding:{param_str}-------")
        config.update_params(
            model_name='TextCNN',
            use_glove=use_glove,
            num_epochs=15
        )
        best_acc, duration=run_training(config)
        results_data.append({
            'group':"Embedding Initialization",
            'param_value':param_str,
            'accuracy':best_acc,
            'duration_sec':duration
        })



    # 实验五：模型架构对比
    print("\n" + "=" * 50)
    print("对比不同模型架构")
    print("=" * 50)
    models_to_test = ['TextCNN', 'TextRNN', 'TextTransformer']
    for model_name in models_to_test:
        print(f"\n------正在测试模型:{model_name}------")
        config = Config()
        config.update_params(model_name=model_name)
        if model_name == 'TextTransformer':
            config.update_params(learning_rate=5e-5)
        else:
            config.update_params(learning_rate=1e-4)
        best_acc, duration = run_training(config)
        results_data.append({
            'group': "Model Architecture",
            'param_value': model_name,
            'accuracy': best_acc,
            'duration_sec': duration
        })
    results_df = pd.DataFrame(results_data)
    if not os.path.exists(os.path.dirname(base_config.result_csv_path)):
        os.makedirs(os.path.dirname(base_config.result_csv_path))
    results_df.to_csv(base_config.result_csv_path, index=False)
    print(f"\nAll experiment results saved to {base_config.result_csv_path}")
    print(results_df)

    plot_all_results(results_df)

if __name__ == "__main__":
    main()