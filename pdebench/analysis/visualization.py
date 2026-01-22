"""
可视化工具 - 从 JSON 数据生成图表

实验 1.1, 4.1, 4.6 的图表生成
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体（如果需要）
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_figure1_pass_rate(data_file: Path, output_file: Path):
    """
    实验 1.1: Figure 1 - Zero-Shot Pass Rate Comparison
    柱状图
    """
    with open(data_file) as f:
        data = json.load(f)
    
    models = data['models']
    pass_rates = data['pass_rates']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, pass_rates, color='steelblue', alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel(data['xlabel'], fontsize=12)
    ax.set_ylabel(data['ylabel'], fontsize=12)
    ax.set_title(data['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: {output_file}")


def plot_figure6_gate_breakdown(data_file: Path, output_file: Path):
    """
    实验 4.1: Figure 6 - Gate Breakdown Pass Rates
    堆叠柱状图
    """
    with open(data_file) as f:
        data = json.load(f)
    
    models = data['models']
    exec_valid = np.array(data['exec_valid_rates'])
    accuracy_pass = np.array(data['accuracy_pass_rates'])
    final_pass = np.array(data['final_pass_rates'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    # 堆叠条形图
    p1 = ax.bar(x, exec_valid, width, label='Exec Valid', color='#4CAF50', alpha=0.8)
    p2 = ax.bar(x, accuracy_pass, width, label='Accuracy Pass', color='#2196F3', alpha=0.8)
    p3 = ax.bar(x, final_pass, width, label='Final Pass (Acc+Time)', color='#FF9800', alpha=0.8)
    
    ax.set_xlabel(data['xlabel'], fontsize=12)
    ax.set_ylabel(data['ylabel'], fontsize=12)
    ax.set_title(data['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: {output_file}")


def plot_figure12_cost_performance(data_file: Path, output_file: Path):
    """
    实验 4.6: Figure 12 - Cost vs Performance Scatter
    散点图，bubble size = latency
    """
    with open(data_file) as f:
        data = json.load(f)
    
    points = data['points']
    
    if not points:
        print(f"⚠️  No cost data available, skipping Figure 12")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for point in points:
        ax.scatter(
            point['total_cost_usd'],
            point['pass_rate_pct'],
            s=point['avg_latency_sec'] * 100,  # bubble size
            alpha=0.6,
            label=point['model']
        )
        
        # 添加标签
        ax.text(
            point['total_cost_usd'],
            point['pass_rate_pct'] + 2,
            point['model'],
            fontsize=9,
            ha='center'
        )
    
    ax.set_xlabel(data['xlabel'], fontsize=12)
    ax.set_ylabel(data['ylabel'], fontsize=12)
    ax.set_title(data['title'], fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    
    # 添加 bubble size 说明
    ax.text(
        0.02, 0.98,
        f"Bubble size ∝ {data['bubble_size_label']}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: {output_file}")


def generate_all_plots(report_dir: Path, output_dir: Path):
    """
    从报告数据生成所有图表
    
    Args:
        report_dir: 包含 JSON 数据文件的目录
        output_dir: 图表输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🎨 Generating Plots from Report Data")
    print(f"{'='*80}\n")
    
    # Figure 1
    fig1_data = report_dir / "figure1_pass_rate.json"
    if fig1_data.exists():
        plot_figure1_pass_rate(fig1_data, output_dir / "figure1_pass_rate.png")
    
    # Figure 6
    fig6_data = report_dir / "figure6_gate_breakdown.json"
    if fig6_data.exists():
        plot_figure6_gate_breakdown(fig6_data, output_dir / "figure6_gate_breakdown.png")
    
    # Figure 12
    fig12_data = report_dir / "figure12_cost_performance.json"
    if fig12_data.exists():
        plot_figure12_cost_performance(fig12_data, output_dir / "figure12_cost_performance.png")
    
    print(f"\n{'='*80}")
    print(f"✅ All plots generated!")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate plots from report JSON data')
    parser.add_argument('--reports', type=Path, required=True, help='Directory containing JSON report files')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for plots')
    args = parser.parse_args()
    
    generate_all_plots(args.reports, args.output)
