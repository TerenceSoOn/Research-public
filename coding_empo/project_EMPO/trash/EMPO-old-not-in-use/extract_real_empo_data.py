#!/usr/bin/env python3
"""
专门针对EMPO TensorBoard数据的图表生成器
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def extract_real_empo_data(tensorboard_dir):
    """从真实的EMPO TensorBoard日志中提取数据，按文件分组"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # 存储每个文件的数据
        file_metrics = {}
        
        # 查找TensorBoard事件文件
        event_files = glob.glob(os.path.join(tensorboard_dir, "**/events.out.tfevents.*"), recursive=True)
        
        if not event_files:
            print(f"No TensorBoard event files found in {tensorboard_dir}")
            return None
        
        print(f"Found {len(event_files)} TensorBoard event files")
        
        for event_file in event_files:
            print(f"\nProcessing: {event_file}")
            
            # 为每个文件创建独立的metrics字典
            metrics = {
                'steps': [],
                'val_acc_mean': [],
                'val_acc_maj': [],
                'val_reward_mean': [],
                'train_loss': [],
                'kl_divergence': [],
                'learning_rate': [],
                'file_path': event_file,
                'file_name': os.path.basename(event_file)
            }
            
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # 获取所有可用的标量标签
            scalar_tags = ea.Tags()['scalars']
            print(f"Total tags found: {len(scalar_tags)}")
            
            # 找到关键指标 - 扩展搜索模式
            acc_tags = [tag for tag in scalar_tags if 'acc' in tag.lower() or 'accuracy' in tag.lower()]
            reward_tags = [tag for tag in scalar_tags if 'reward' in tag.lower()]
            loss_tags = [tag for tag in scalar_tags if 'loss' in tag.lower()]
            lr_tags = [tag for tag in scalar_tags if 'lr' in tag.lower() or 'learning_rate' in tag.lower()]
            entropy_tags = [tag for tag in scalar_tags if 'entropy' in tag.lower()]
            
            print(f"Accuracy tags: {acc_tags[:5]}...")  # 显示前5个
            print(f"Reward tags: {reward_tags[:5]}...")
            print(f"Loss tags: {loss_tags[:5]}...")
            
            # 主要关注验证准确率 - 更灵活的匹配，包括奖励作为替代指标
            main_acc_tag = None
            for tag in acc_tags:
                if any(pattern in tag.lower() for pattern in ['mean@16', 'maj@16', 'val_acc', 'accuracy']):
                    main_acc_tag = tag
                    break
            
            # 如果没找到accuracy标签，使用reward标签作为主要指标
            if not main_acc_tag and reward_tags:
                main_acc_tag = reward_tags[0]  # 使用第一个reward标签
                print(f"No accuracy tags found, using reward tag as primary metric: {main_acc_tag}")
            elif not main_acc_tag and acc_tags:
                main_acc_tag = acc_tags[0]
            
            if main_acc_tag:
                print(f"Using main accuracy tag: {main_acc_tag}")
                scalar_events = ea.Scalars(main_acc_tag)
                print(f"Found {len(scalar_events)} events for main tag")
                
                for event in scalar_events:
                    step = event.step
                    value = event.value
                    metrics['steps'].append(step)
                    metrics['val_acc_mean'].append(value)
            
            # 提取其他指标
            for key in ['val_acc_maj', 'val_reward_mean', 'train_loss', 'kl_divergence', 'learning_rate']:
                metrics[key] = [None] * len(metrics['steps'])
            
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                for event in scalar_events:
                    step = event.step
                    value = event.value
                    
                    if step in metrics['steps']:
                        idx = metrics['steps'].index(step)
                        
                        # 分类指标 - 更灵活的匹配
                        if any(pattern in tag.lower() for pattern in ['acc', 'accuracy']) and any(pattern in tag.lower() for pattern in ['maj@16', 'majority']):
                            metrics['val_acc_maj'][idx] = value
                        elif any(pattern in tag.lower() for pattern in ['reward']) and any(pattern in tag.lower() for pattern in ['mean', 'val', 'validation']):
                            metrics['val_reward_mean'][idx] = value
                        elif any(pattern in tag.lower() for pattern in ['pg_loss', 'policy_loss', 'actor_loss', 'loss']):
                            metrics['train_loss'][idx] = value
                        elif any(pattern in tag.lower() for pattern in ['ppo_kl', 'kl_div', 'kl']):
                            metrics['kl_divergence'][idx] = value
                        elif any(pattern in tag.lower() for pattern in ['actor/lr', 'learning_rate', 'lr']):
                            metrics['learning_rate'][idx] = value
            
            # 排序数据
            if metrics['steps']:
                sorted_indices = sorted(range(len(metrics['steps'])), key=lambda i: metrics['steps'][i])
                for key in metrics:
                    if key not in ['file_path', 'file_name'] and isinstance(metrics[key], list):
                        metrics[key] = [metrics[key][i] for i in sorted_indices]
                
                print(f"File {os.path.basename(event_file)}: {len(metrics['steps'])} steps, range {min(metrics['steps'])}-{max(metrics['steps'])}")
                file_metrics[event_file] = metrics
        
        return file_metrics
        
    except ImportError:
        print("TensorBoard not available. Please install: pip install tensorboard")
        return None
    except Exception as e:
        print(f"Error processing TensorBoard logs: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_real_empo_data(file_metrics, output_dir):
    """绘制真实的EMPO训练数据，每个文件使用不同颜色"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EMPO Training Progress', fontsize=16)
    
    # 定义颜色循环
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*']
    
    file_list = list(file_metrics.keys())
    print(f"\nPlotting data from {len(file_list)} files:")
    
    legend_labels = []
    
    for i, (event_file, metrics) in enumerate(file_metrics.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # 创建标签，包含文件时间信息
        file_name = os.path.basename(event_file)
        # 从文件名中提取时间戳
        if 'tfevents' in file_name:
            timestamp_part = file_name.split('.')[-2] if '.' in file_name else file_name
            label = f"Session {i+1} ({timestamp_part[:10]})"
        else:
            label = f"Session {i+1}"
        
        legend_labels.append(label)
        steps = metrics['steps']
        
        print(f"  {label}: {len(steps)} steps, range {min(steps) if steps else 'N/A'}-{max(steps) if steps else 'N/A'}")
        
        # 1. 验证准确率
        if metrics['val_acc_mean']:
            val_acc_clean = [(s, a) for s, a in zip(steps, metrics['val_acc_mean']) if a is not None]
            if val_acc_clean:
                steps_clean, acc_clean = zip(*val_acc_clean)
                axes[0, 0].plot(steps_clean, acc_clean, color=color, marker=marker, 
                              linewidth=2, markersize=4, label=label, alpha=0.8)
                axes[0, 0].set_title('Validation Accuracy (Mean@16)', fontsize=14)
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 验证奖励
        if metrics['val_reward_mean']:
            val_reward_clean = [(s, r) for s, r in zip(steps, metrics['val_reward_mean']) if r is not None]
            if val_reward_clean:
                steps_clean, reward_clean = zip(*val_reward_clean)
                axes[0, 1].plot(steps_clean, reward_clean, color=color, marker=marker,
                              linewidth=2, markersize=4, label=label, alpha=0.8)
                axes[0, 1].set_title('Validation Reward', fontsize=14)
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('Reward')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练损失
        if metrics['train_loss']:
            train_loss_clean = [(s, l) for s, l in zip(steps, metrics['train_loss']) if l is not None]
            if train_loss_clean:
                steps_clean, loss_clean = zip(*train_loss_clean)
                axes[1, 0].plot(steps_clean, loss_clean, color=color, marker=marker,
                              linewidth=2, markersize=4, label=label, alpha=0.8)
                axes[1, 0].set_title('Training Loss', fontsize=14)
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. KL散度
        if metrics['kl_divergence']:
            kl_clean = [(s, k) for s, k in zip(steps, metrics['kl_divergence']) if k is not None]
            if kl_clean:
                steps_clean, kl_div_clean = zip(*kl_clean)
                axes[1, 1].plot(steps_clean, kl_div_clean, color=color, marker=marker,
                              linewidth=2, markersize=4, label=label, alpha=0.8)
                axes[1, 1].set_title('KL Divergence', fontsize=14)
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('KL Divergence')
                axes[1, 1].grid(True, alpha=0.3)
    
    # 为每个子图添加图例
    for ax in axes.flat:
        if ax.get_lines():  # 只有当子图有数据时才添加图例
            ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'real_empo_training_progress_multi_session.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-session EMPO training chart saved to: {output_path}")
    
    return output_path

# ///////////////////////////////////////////////////////////////////
def process_tensorboard_directories():
    """处理多个TensorBoard目录"""
    # 定义所有可能的TensorBoard日志目录
    tensorboard_dirs = [
        "//data/coding/EMPO/verl-empo/tensorboard_log"
    ]
    
    output_base_dir = "/data/coding/EMPO-main/real_validation_plots"
    os.makedirs(output_base_dir, exist_ok=True)
    
    generated_plots = []
    
    for i, tensorboard_dir in enumerate(tensorboard_dirs):
        if not os.path.exists(tensorboard_dir):
            print(f"Directory does not exist: {tensorboard_dir}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing directory {i+1}/{len(tensorboard_dirs)}: {tensorboard_dir}")
        print(f"{'='*60}")
        
        # 为每个目录创建单独的输出文件夹
        dir_name = os.path.basename(tensorboard_dir) or f"experiment_{i+1}"
        output_dir = os.path.join(output_base_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print("Extracting EMPO data from TensorBoard...")
        file_metrics = extract_real_empo_data(tensorboard_dir)
        
        if file_metrics and len(file_metrics) > 0:
            total_steps = sum(len(metrics['steps']) for metrics in file_metrics.values())
            print(f"Found {len(file_metrics)} TensorBoard files with {total_steps} total training steps")
            
            # 显示每个文件的验证准确率
            print(f"\n{'='*50}")
            print("VALIDATION ACCURACY SUMMARY:")
            print(f"{'='*50}")
            
            for event_file, metrics in file_metrics.items():
                if metrics['steps']:
                    file_name = os.path.basename(event_file)
                    print(f"\nFile: {file_name}")
                    print(f"Steps: {len(metrics['steps'])}, Range: {min(metrics['steps'])}-{max(metrics['steps'])}")
                    
                    # 显示验证准确率
                    if metrics['val_acc_mean']:
                        val_acc_clean = [(s, a) for s, a in zip(metrics['steps'], metrics['val_acc_mean']) if a is not None]
                        if val_acc_clean:
                            print(f"Validation Accuracy (Mean@16):")
                            for step, acc in val_acc_clean:
                                print(f"  Step {step:4d}: {acc:.4f} ({acc*100:.2f}%)")
            
            print(f"\n{'='*50}")
            
            plot_path = plot_real_empo_data(file_metrics, output_dir)
            print(f"Chart generated: {plot_path}")
            generated_plots.append(plot_path)
        else:
            print("No valid data found in TensorBoard logs")
    
    return generated_plots

def main():
    plots = process_tensorboard_directories()
    
    if plots:
        print(f"\n{'='*60}")
        print(f"Successfully generated {len(plots)} training progress charts:")
        for plot in plots:
            print(f"  - {plot}")
        print(f"{'='*60}")
    else:
        print("No charts were generated. Please check if TensorBoard logs exist.")

if __name__ == "__main__":
    main()
