import matplotlib.pyplot as plt
import numpy as np

def plot_kv_cache_edge_focused():
    # --- 1. 模型参数 (Qwen-4B) ---
    n_layers = 36
    hidden_size = 2560
    gqa = 4 
    bytes_per_param = 2  # FP16/BF16
    
    # 估算模型权重本身的大小 (4B * 2 Bytes = 8GB)
    # 实际上还有一些 overhead，这里取 8GB 作为基准
    model_weights_gb = 4 * 10**9 * 2 / (1024**3) # approx 7.45GB, let's say ~8GB for easy visual
    model_weights_gb_visual = 8.0 

    design_limit = 128 * 1024

    # --- 2. KV Cache 计算公式 ---
    def calc_kv_gb(seq_len):
        bytes_per_token = 2 * n_layers * (hidden_size / gqa) * bytes_per_param
        total_bytes = bytes_per_token * seq_len
        return total_bytes / (1024**3)

    # --- 3. 序列长度 (Log Scale) ---
    seq_lens = np.geomspace(1024, 1024*1024, 500)
    
    # 计算总显存占用 = KV Cache + 模型权重 (这是端侧用户最关心的)
    kv_sizes = calc_kv_gb(seq_lens)
    total_mem_usage = kv_sizes + model_weights_gb_visual

    # --- 4. 绘图设置 ---
    plt.figure(figsize=(12, 7.5), dpi=150)
    
    # --- 核心改动：绘制模型权重的基准区域 ---
    # 在端侧，权重的占用是不可忽视的
    plt.axhspan(0, model_weights_gb_visual, color='#cccccc', alpha=0.3, label='Model Weights (~8GB)')
    plt.text(1200, 4, 'Model Weights Area\n(FP16)', color='#666666', fontsize=10, fontweight='bold', va='center')
    plt.text(270*1024, 10, 'KV Cache Area\n(FP16)', color='#E77500', fontsize=10, fontweight='bold', va='center')
    # 绘制 KV Cache 增长曲线 (叠加在权重之上)
    plt.plot(seq_lens, total_mem_usage, label='Total Memory Usage (Weights + KV)', color='#E77500', linewidth=3)
    plt.fill_between(seq_lens, model_weights_gb_visual, total_mem_usage, color='#E77500', alpha=0.1)

    # --- 5. 设置 X 轴 (Log2) ---
    plt.xscale('log', base=2)
    ticks = [1024 * (2**i) for i in range(11)]
    tick_labels = ['1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M']
    plt.xticks(ticks, tick_labels, fontsize=10)

    # --- 6. 端侧显卡参考线 (重点更新) ---
    gpu_specs = [
        # 端侧/消费级设备
        ('MacBook Air / Entry Laptop (8GB)', 8, '#333333'),  # 8GB 几乎不可用
        ('High-end Android / RTX 3060 (12GB)', 12, 'teal'),
        ('MacBook Pro / RTX 4060 Ti (16GB)', 16, 'purple'),
        ('RTX 3090 / 4090 (24GB)', 24, 'green'),
        
        # 数据中心卡 (保留一个作为对比，稍微淡化)
        ('A100 (40GB)', 40, 'blue'),
    ]
    
    # 稍微错开文字位置防止重叠
    text_y_offsets = {8: -1.2, 12: 0.5, 16: 0.5, 24: 0.5, 40: 0.5} 

    for name, mem, color in gpu_specs:
        style = '--' if mem < 30 else '-.' # 端侧用虚线，服务器用点划线
        width = 1.5 if mem < 30 else 1.0
        alpha = 0.8 if mem < 30 else 0.4
        
        plt.axhline(y=mem, color=color, linestyle=style, alpha=alpha, linewidth=width)
        plt.text(1024, mem + text_y_offsets.get(mem, 0.5), f'  {name}', 
                 color=color, fontweight='bold', fontsize=10, va='bottom')

    # --- 7. 设计极限标记 ---
    plt.axvline(x=design_limit, color='#333333', linestyle=':', linewidth=2)
    # 计算在 128k 时的总显存
    limit_mem = calc_kv_gb(design_limit) + model_weights_gb_visual
    plt.plot(design_limit, limit_mem, 'o', color='#333333')
    plt.text(design_limit * 1.1, limit_mem, 
             f'Design Limit (128k)\nRequires ~{limit_mem:.1f} GB', 
             color='#333333', fontsize=11, va='center')

    # --- 8. 标题和标签 ---
    plt.title('Edge Device Memory Bottleneck: Qwen-4B (FP16)', fontsize=16, pad=20)
    plt.xlabel('Context Length (Tokens)', fontsize=14)
    plt.ylabel('Total Memory Usage (GB)', fontsize=14)
    plt.ylim(0, 45) # 限制 Y 轴高度，专注于端侧范围 (0-45GB)
    
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.legend(loc='lower right', framealpha=1)
    plt.tight_layout()
    
    plt.savefig('kv_cache_edge_device.png')
    plt.show()

if __name__ == "__main__":
    plot_kv_cache_edge_focused()