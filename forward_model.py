import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class EEGForwardModel:
    """EEG源定位正问题模拟"""
    
    def __init__(self, n_channels=64, n_sources=5000, head_radius=0.09):
        """
        初始化EEG正向模型
        
        Parameters:
        -----------
        n_channels: int, 头皮电极数量
        n_sources: int, 源空间网格点数量（决策变量维度）
        head_radius: float, 头模型半径（米）
        """
        self.n_channels = n_channels
        self.n_sources = n_sources
        self.head_radius = head_radius
        
        # 生成电极位置（头皮表面）
        self.sensor_pos = self._generate_sensor_positions()
        
        # 生成源空间位置（大脑皮层）
        self.source_pos = self._generate_source_positions()
        
        # 计算导联场矩阵
        self.leadfield = self._compute_leadfield()
        
    def _generate_sensor_positions(self):
        """生成头皮电极位置（球面均匀分布）"""
        indices = np.arange(0, self.n_channels, dtype=float) + 0.5
        
        # 使用黄金螺旋算法生成球面均匀点
        phi = np.arccos(1 - 2*indices/self.n_channels)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = self.head_radius * np.cos(theta) * np.sin(phi)
        y = self.head_radius * np.sin(theta) * np.sin(phi)
        z = self.head_radius * np.cos(phi)
        
        return np.column_stack([x, y, z])
    
    def _generate_source_positions(self):
        """生成源空间位置（内球面，模拟大脑皮层）"""
        indices = np.arange(0, self.n_sources, dtype=float) + 0.5
        
        # 源空间在内部，半径约为头皮的70%
        source_radius = self.head_radius * 0.7
        
        phi = np.arccos(1 - 2*indices/self.n_sources)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = source_radius * np.cos(theta) * np.sin(phi)
        y = source_radius * np.sin(theta) * np.sin(phi)
        z = source_radius * np.cos(phi)
        
        return np.column_stack([x, y, z])
    
    def _compute_leadfield(self):
        """
        计算导联场矩阵（使用简化的球形头模型）
        基于三层球模型的解析解
        """
        print("正在计算导联场矩阵...")
        
        # 计算每个源点到每个电极的距离
        distances = cdist(self.sensor_pos, self.source_pos)
        
        # 简化的导联场计算（基于距离的反比）
        # 实际应用中会使用更精确的边界元法(BEM)或有限元法(FEM)
        leadfield = 1.0 / (4 * np.pi * distances**2 + 1e-6)
        
        # 归一化
        leadfield = leadfield / np.max(np.abs(leadfield))
        
        print(f"导联场矩阵维度: {leadfield.shape}")
        print(f"条件数: {np.linalg.cond(leadfield):.2e} (病态矩阵，需要正则化)")
        
        return leadfield
    
    def create_point_source(self, location_idx, amplitude=1.0):
        """
        创建点源信号
        
        Parameters:
        -----------
        location_idx: int or list, 源位置索引
        amplitude: float or list, 源强度
        
        Returns:
        --------
        source_signal: ndarray, shape (n_sources,)
        """
        source_signal = np.zeros(self.n_sources)
        
        if isinstance(location_idx, (list, np.ndarray)):
            for idx, amp in zip(location_idx, amplitude):
                source_signal[idx] = amp
        else:
            source_signal[location_idx] = amplitude
            
        return source_signal
    
    def create_distributed_source(self, center_idx, width=50, amplitude=1.0):
        """
        创建分布式源（高斯分布）
        
        Parameters:
        -----------
        center_idx: int, 中心位置索引
        width: float, 分布宽度（源点数）
        amplitude: float, 峰值强度
        """
        source_signal = np.zeros(self.n_sources)
        
        # 计算所有源点到中心的距离
        center_pos = self.source_pos[center_idx]
        distances = np.linalg.norm(self.source_pos - center_pos, axis=1)
        
        # 高斯分布
        sigma = width * (self.source_pos[1, 0] - self.source_pos[0, 0])  # 估计网格间距
        source_signal = amplitude * np.exp(-(distances**2) / (2 * sigma**2))
        
        return source_signal
    
    def forward_solution(self, source_signal, noise_level=0.1):
        """
        正向求解：从源信号计算头皮电位
        
        Parameters:
        -----------
        source_signal: ndarray, shape (n_sources,), 源空间电流密度
        noise_level: float, 噪声水平（信噪比）
        
        Returns:
        --------
        scalp_potential: ndarray, shape (n_channels,), 头皮电位测量
        """
        # 正向计算: y = Lx
        scalp_potential = self.leadfield @ source_signal
        
        # 添加测量噪声
        noise = noise_level * np.random.randn(self.n_channels)
        scalp_potential += noise * np.std(scalp_potential)
        
        return scalp_potential
    
    def visualize_setup(self):
        """可视化电极和源空间分布"""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D视图
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(self.sensor_pos[:, 0], self.sensor_pos[:, 1], 
                   self.sensor_pos[:, 2], c='red', s=50, label='电极')
        ax1.scatter(self.source_pos[:, 0], self.source_pos[:, 1], 
                   self.source_pos[:, 2], c='blue', s=1, alpha=0.3, label='源空间')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('头模型三维视图')
        ax1.legend()
        
        # 顶视图
        ax2 = fig.add_subplot(132)
        ax2.scatter(self.sensor_pos[:, 0], self.sensor_pos[:, 1], 
                   c='red', s=50, label='电极')
        ax2.scatter(self.source_pos[:, 0], self.source_pos[:, 1], 
                   c='blue', s=1, alpha=0.3, label='源空间')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('顶视图')
        ax2.axis('equal')
        ax2.legend()
        
        # 导联场矩阵可视化
        ax3 = fig.add_subplot(133)
        im = ax3.imshow(self.leadfield, aspect='auto', cmap='RdBu_r')
        ax3.set_xlabel('源空间索引')
        ax3.set_ylabel('电极索引')
        ax3.set_title(f'导联场矩阵 L ({self.n_channels}×{self.n_sources})')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        plt.show()


# ===== 使用示例 =====
if __name__ == "__main__":
    # 1. 创建正向模型
    print("="*60)
    print("创建EEG正向模型")
    print("="*60)
    model = EEGForwardModel(n_channels=64, n_sources=10000)
    
    # 可视化模型设置
    model.visualize_setup()
    
    # 2. 测试用例1：单点源
    print("\n" + "="*60)
    print("测试用例1: 单点源")
    print("="*60)
    source_idx = 5000  # 选择一个源位置
    x_true = model.create_point_source(source_idx, amplitude=1.0)
    y_measured = model.forward_solution(x_true, noise_level=0.1)
    
    print(f"源信号非零元素数: {np.sum(x_true != 0)}")
    print(f"头皮电位范围: [{y_measured.min():.4f}, {y_measured.max():.4f}]")
    
    # 3. 测试用例2：双点源
    print("\n" + "="*60)
    print("测试用例2: 双点源")
    print("="*60)
    x_true2 = model.create_point_source([3000, 7000], amplitude=[1.0, 0.8])
    y_measured2 = model.forward_solution(x_true2, noise_level=0.1)
    
    print(f"源信号非零元素数: {np.sum(x_true2 != 0)}")
    print(f"头皮电位范围: [{y_measured2.min():.4f}, {y_measured2.max():.4f}]")
    
    # 4. 测试用例3：分布式源
    print("\n" + "="*60)
    print("测试用例3: 分布式源（高斯分布）")
    print("="*60)
    x_true3 = model.create_distributed_source(center_idx=5000, width=100, amplitude=1.0)
    y_measured3 = model.forward_solution(x_true3, noise_level=0.1)
    
    print(f"源信号非零元素数: {np.sum(x_true3 > 0.01)}")
    print(f"头皮电位范围: [{y_measured3.min():.4f}, {y_measured3.max():.4f}]")
    
    # 5. 可视化结果
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    test_cases = [
        (x_true, y_measured, "单点源"),
        (x_true2, y_measured2, "双点源"),
        (x_true3, y_measured3, "分布式源")
    ]
    
    for i, (x, y, title) in enumerate(test_cases):
        # 源信号
        axes[i, 0].plot(x)
        axes[i, 0].set_xlabel('源空间索引')
        axes[i, 0].set_ylabel('电流密度')
        axes[i, 0].set_title(f'{title} - 源信号')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 头皮电位
        axes[i, 1].plot(y, 'o-')
        axes[i, 1].set_xlabel('电极索引')
        axes[i, 1].set_ylabel('电位 (归一化)')
        axes[i, 1].set_title(f'{title} - 头皮电位')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("正向模型构建完成！")
    print(f"下一步：实现逆问题求解算法")
    print("="*60)