import numpy as np
from typing import Optional
import mujoco


class TerrainSubComponent:
    """程序化地形生成与管理组件"""
    
    def __init__(self, config):
        self.config = config
        self.terrain_type = config.mesh_type
        self.nrow = 200
        self.ncol = 200
        self.size_x = 10.0
        self.size_y = 10.0
        self.current_height_field = None
        
    # def generate_terrain(self, terrain_type: Optional[str] = None) -> np.ndarray:
    #     """生成高度场数据"""
    #     if terrain_type is None:
    #         terrain_type = self.terrain_type
            
    #     nrow, ncol = self.nrow, self.ncol
    #     size_x, size_y = self.size_x, self.size_y
        
    #     if terrain_type == "plane":
    #         height = np.zeros((nrow, ncol))
            
    #     elif terrain_type == "slope":
    #         angle = np.radians(self.config.get("slope_angle", 20.0))
    #         x = np.linspace(-size_x, size_x, ncol)
    #         slope_height = np.tan(angle) * (x + size_x) / (2 * size_x) * 0.4
    #         height = np.tile(slope_height, (nrow, 1))
            
    #     elif terrain_type == "stairs":
    #         # stair_height = self.config.get("stair_height", 0.10)
    #         # x = np.linspace(-size_x, size_x, ncol)
    #         # step_width = 0.5
    #         # stair_steps = np.floor((x + size_x) / step_width) * stair_height
    #         # stair_steps = np.clip(stair_steps, 0, 0.5)
    #         # height = np.tile(stair_steps, (nrow, 1))

    #         # stair_height = self.config.get("stair_height", 0.10)
    #         # step_width = 0.5
    #         # max_height = 0.5
    #         # x = np.linspace(-size_x, size_x, ncol)  # x从 -10 到 10
    #         # shift = 5.0  # 左移5米，使台阶从x=-5开始
    #         # stair_steps = np.floor((x - shift + size_x) / step_width) * stair_height
    #         # stair_steps = np.clip(stair_steps, 0, max_height)
    #         # height = np.tile(stair_steps, (nrow, 1))

    #         stair_height = self.config.get("stair_height", 0.10)
    #         step_width = 0.5
    #         max_height = 0.5
    #         base_height = 0.1  # 基础抬高值
    #         x = np.linspace(-size_x, size_x, ncol)  # x从 -10 到 10
    #         shift = 5.0  # 左移5米，使台阶从x=-5开始
    #         # 计算台阶高度（从0开始）
    #         stair_steps = np.floor((x - shift + size_x) / step_width) * stair_height
    #         stair_steps = np.clip(stair_steps, 0, max_height)
    #         # 整体抬高 base_height
    #         stair_steps = stair_steps + base_height
    #         height = np.tile(stair_steps, (nrow, 1))
            
    #     elif terrain_type == "rough":
    #         rough_range = self.config.get("rough_range", (-0.05, 0.05))
    #         height = np.random.uniform(rough_range[0], rough_range[1], (nrow, ncol))
            
    #     elif terrain_type == "mixed":
    #         height = np.zeros((nrow, ncol))
    #         half_r, half_c = nrow // 2, ncol // 2
            
    #         # 左下：平地
    #         height[:half_r, :half_c] = 0.0
            
    #         # 右下：斜坡 (20°)
    #         angle = np.radians(20.0)
    #         x_slope = np.linspace(0, size_x, half_c)
    #         slope_h = np.tan(angle) * (x_slope / size_x) * 0.4
    #         height[:half_r, half_c:] = np.tile(slope_h, (half_r, 1))
            
    #         # 左上：台阶 (10cm)
    #         x_stair = np.linspace(-size_x, 0, half_c)
    #         stair_h = np.floor((x_stair + size_x) / 0.5) * 0.10
    #         stair_h = np.clip(stair_h, 0, 0.5)
    #         height[half_r:, :half_c] = np.tile(stair_h, (half_r, 1))
            
    #         # 右上：随机凹凸 (±5cm)
    #         height[half_r:, half_c:] = np.random.uniform(-0.05, 0.05, (half_r, half_c))
            
    #     else:
    #         raise ValueError(f"Unknown terrain type: {terrain_type}")
            
    #     self.current_height_field = height
    #     return height
    def generate_terrain(self, terrain_type: Optional[str] = None) -> np.ndarray:
        """生成高度场数据"""
        if terrain_type is None:
            terrain_type = self.terrain_type
            
        nrow, ncol = self.nrow, self.ncol
        size_x, size_y = self.size_x, self.size_y
        
        # 基础抬升值，防止穿模
        base_height = 0.1
        
        if terrain_type == "plane":
            height = np.ones((nrow, ncol)) * base_height
            
        elif terrain_type == "slope":
            nrow, ncol = self.nrow, self.ncol
            size_x, size_y = self.size_x, self.size_y
            base_height = 0.1
            angle = np.radians(self.config.get("slope_angle", 20.0))
            
            x = np.linspace(-1, 1, ncol)
            y = np.linspace(-1, 1, nrow)
            X, Y = np.meshgrid(x, y)
            dist = np.maximum(np.abs(X), np.abs(Y))
            
            # 中心平台半径 (归一化)
            platform_radius = 1.0 / size_x  # 0.1
            
            # 转换为物理距离 (乘以 size_x)
            physical_dist = dist * size_x  # 0 ~ 10m
            
            # 中间低，四周高，用物理距离计算
            height = base_height + np.tan(angle) * (physical_dist - 1.0)  # 平台边缘在 1m
            height[dist <= platform_radius] = base_height
            height = np.clip(height, base_height, None)
            
        # elif terrain_type == "stairs":
        #     stair_height = self.config.get("stair_height", 0.10)
        #     step_width = 0.5
        #     max_height = 0.5
        #     x = np.linspace(-size_x, size_x, ncol)  # x从 -10 到 10
        #     shift = 5.0  # 左移5米，使台阶从x=-5开始
        #     # 计算台阶高度（从0开始）
        #     stair_steps = np.floor((x - shift + size_x) / step_width) * stair_height
        #     stair_steps = np.clip(stair_steps, 0, max_height)
        #     # 整体抬高 base_height
        #     stair_steps = stair_steps + base_height
        #     height = np.tile(stair_steps, (nrow, 1))
        # elif terrain_type == "stairs":
        #     nrow, ncol = self.nrow, self.ncol
        #     size_x, size_y = self.size_x, self.size_y
        #     base_height = 0.1
        #     max_height = 0.6
        #     step_height = 0.1
        #     step_width = 0.5
            
        #     x = np.linspace(-1, 1, ncol)
        #     y = np.linspace(-1, 1, nrow)
        #     X, Y = np.meshgrid(x, y)
        #     dist = np.maximum(np.abs(X), np.abs(Y))
            
        #     # 中心平台半径 (2x2 平台，半边长1m)
        #     platform_radius = 1.0 / size_x  # 0.1
        #     step_width_norm = step_width / size_x  # 0.05
            
        #     # 从中心向外计算层级
        #     # dist 越大，层级越高，高度越低
        #     level = np.floor((dist - platform_radius) / step_width_norm) + 1
        #     level = np.clip(level, 0, 5)
            
        #     # 高度 = max_height - level * step_height
        #     # level=0 → 0.6 (中心), level=1 → 0.5, ..., level=5 → 0.1
        #     height = max_height - level * step_height
        #     height = np.clip(height, base_height, max_height)
            
        #     # 中心平台强制为 max_height (0.6)
        #     height[dist <= platform_radius] = max_height
        elif terrain_type == "stairs":
            nrow, ncol = self.nrow, self.ncol
            size_x, size_y = self.size_x, self.size_y
            base_height = 0.1
            max_height = 0.6
            step_height = self.config.get("stair_height", 0.10)
            step_width = 0.5
            
            x = np.linspace(-1, 1, ncol)
            y = np.linspace(-1, 1, nrow)
            X, Y = np.meshgrid(x, y)
            dist = np.maximum(np.abs(X), np.abs(Y))
            
            # 中心平台半径 (2x2 平台，半边长1m)
            platform_radius = 1.0 / size_x  # 0.1
            step_width_norm = step_width / size_x  # 0.05
            
            # ===== 修改：中间低，四周高 =====
            # 从中心向外计算层级
            # dist 越大，层级越高，高度越高（反过来了）
            level = np.floor((dist - platform_radius) / step_width_norm) + 1
            level = np.clip(level, 0, 5)
            
            # 高度 = base_height + level * step_height
            # level=0 → 0.1 (中心最低), level=1 → 0.2, ..., level=5 → 0.6 (最外圈最高)
            height = base_height + level * step_height
            height = np.clip(height, base_height, max_height)
            
            # 中心平台强制为 base_height (0.1) 最低
            height[dist <= platform_radius] = base_height
            
        elif terrain_type == "rough":
            rough_range = self.config.get("rough_range", (-0.05, 0.05))
            height = np.random.uniform(rough_range[0], rough_range[1], (nrow, ncol)) + base_height
            
        # elif terrain_type == "mixed":
        #     height = np.zeros((nrow, ncol))
        #     half_r, half_c = nrow // 2, ncol // 2
            
        #     # 左下：平地
        #     height[:half_r, :half_c] = base_height
            
        #     # 右下：斜坡 (20°)
        #     angle = np.radians(20.0)
        #     x_slope = np.linspace(0, size_x, half_c)
        #     slope_h = np.tan(angle) * (x_slope / size_x) * 0.4 + base_height
        #     height[:half_r, half_c:] = np.tile(slope_h, (half_r, 1))
            
        #     # 左上：台阶 (10cm)
        #     x_stair = np.linspace(-size_x, 0, half_c)
        #     stair_h = np.floor((x_stair + size_x) / 0.5) * 0.10
        #     stair_h = np.clip(stair_h, 0, 0.5) + base_height
        #     height[half_r:, :half_c] = np.tile(stair_h, (half_r, 1))
            
        #     # 右上：随机凹凸 (±5cm)
        #     height[half_r:, half_c:] = np.random.uniform(-0.05, 0.05, (half_r, half_c)) + base_height
        # elif terrain_type == "mixed":
        #     height = np.zeros((nrow, ncol))
        #     half_r, half_c = nrow // 2, ncol // 2
            
        #     # 左下：平地
        #     height[:half_r, :half_c] = base_height
            
        #     # 右下：斜坡 (20°)
        #     angle = np.radians(20.0)
        #     x_slope = np.linspace(0, size_x, half_c)
        #     slope_h = np.tan(angle) * (x_slope / size_x) * 0.4 + base_height
        #     height[:half_r, half_c:] = np.tile(slope_h, (half_r, 1))
            
        #     # 左上：金字塔式台阶 (中间最高，向四周降低)
        #     # 获取台阶区域的网格坐标 (x: -10 到 0, y: 0 到 10)
        #     x_stair = np.linspace(-size_x, 0, half_c)
        #     y_stair = np.linspace(0, size_y, half_r)
        #     X_stair, Y_stair = np.meshgrid(x_stair, y_stair)
            
        #     # 归一化到 [-1, 1]，中心在台阶区域中心 (-5, 5)
        #     x_norm = (X_stair + 5) / 5   # -10~0 → -1~1, 中心 -5 → 0
        #     y_norm = (Y_stair - 5) / 5   # 0~10 → -1~1, 中心 5 → 0
            
        #     # 切比雪夫距离 (方形等高层)
        #     dist = np.maximum(np.abs(x_norm), np.abs(y_norm))
            
        #     # 方案一：金字塔式高度
        #     # 中心平台半径 (2x2 平台，半边长1m)
        #     # 注意：现在 1m 对应归一化距离 1/5 = 0.2 (因为半边长5m)
        #     platform_radius = 1.0 / 5   # 0.2 (2x2平台，半边长1m，归一化到 [-1,1] 范围)
        #     step_width_norm = 0.5 / 5   # 0.1 (每级台阶宽度0.5m)
        #     max_height = 0.5 + base_height  # 0.6
        #     step_height = 0.1
            
        #     # 从中心向外计算层级
        #     level = np.floor((dist - platform_radius) / step_width_norm) + 1
        #     level = np.clip(level, 0, 5)
            
        #     # 高度 = max_height - level * step_height
        #     stair_h = max_height - level * step_height
        #     stair_h = np.clip(stair_h, base_height, max_height)
            
        #     # 中心平台强制为 max_height
        #     stair_h[dist <= platform_radius] = max_height
            
        #     # 应用到台阶区域
        #     height[half_r:, :half_c] = stair_h
            
        #     # 右上：随机凹凸 (±5cm)
        #     height[half_r:, half_c:] = np.random.uniform(-0.05, 0.05, (half_r, half_c)) + base_height
        elif terrain_type == "mixed":
            height = np.zeros((nrow, ncol))
            half_r, half_c = nrow // 2, ncol // 2
            
            # 左下：平地
            height[:half_r, :half_c] = 0.55
            
            # 右下：碗形斜坡 (中间低，四周高)
            angle = np.radians(self.config.get("slope_angle", 20.0))
            # 获取右下区域的网格坐标 (x: 0~10, y: -10~0)
            x_bowl = np.linspace(0, size_x, half_c)
            y_bowl = np.linspace(-size_y, 0, half_r)
            X_bowl, Y_bowl = np.meshgrid(x_bowl, y_bowl)
            
            # 归一化到 [-1, 1]，中心在 (5, -5)
            x_norm = (X_bowl - 5) / 5
            y_norm = (Y_bowl + 5) / 5
            dist = np.maximum(np.abs(x_norm), np.abs(y_norm))
            
            # 中心平台半径 (2x2 平台)
            platform_radius = 1.0 / 5  # 0.2
            
            # 转换为物理距离 (乘以半边长 5m)
            physical_dist = dist * 5
            
            # 中间低，四周高：从平台边缘开始升高
            bowl_h = np.tan(angle) * (physical_dist - 1.0)  # 平台边缘在 1m
            bowl_h = np.clip(bowl_h, 0, None)
            bowl_h[dist <= platform_radius] = 0  # 中心 2x2 平台最低
            height[:half_r, half_c:] = bowl_h + base_height
            
            # 左上：倒金字塔式台阶 (中间最低，向四周升高)
            # 获取台阶区域的网格坐标 (x: -10 到 0, y: 0 到 10)
            x_stair = np.linspace(-size_x, 0, half_c)
            y_stair = np.linspace(0, size_y, half_r)
            X_stair, Y_stair = np.meshgrid(x_stair, y_stair)
            
            # 归一化到 [-1, 1]，中心在台阶区域中心 (-5, 5)
            x_norm = (X_stair + 5) / 5   # -10~0 → -1~1, 中心 -5 → 0
            y_norm = (Y_stair - 5) / 5   # 0~10 → -1~1, 中心 5 → 0
            
            # 切比雪夫距离 (方形等高层)
            dist = np.maximum(np.abs(x_norm), np.abs(y_norm))
            
            # ===== 修改：中间低，四周高 =====
            platform_radius = 1.0 / 5   # 0.2
            step_width_norm = 0.5 / 5   # 0.1
            max_height = 0.5 + base_height  # 0.6
            step_height = self.config.get("stair_height", 0.10)
            
            # 从中心向外计算层级
            level = np.floor((dist - platform_radius) / step_width_norm) + 1
            level = np.clip(level, 0, 10)
            
            # 高度 = base_height + level * step_height
            # level=0 → 0.1 (中心最低), level=5 → 0.6 (最外圈最高)
            stair_h = base_height + level * step_height
            stair_h = np.clip(stair_h, base_height, max_height)
            
            # 中心平台强制为 base_height (0.1) 最低
            stair_h[dist <= platform_radius] = base_height
            
            # 应用到台阶区域
            height[half_r:, :half_c] = stair_h
            
            # 右上：随机凹凸 (±5cm)
            height[half_r:, half_c:] = np.random.uniform(-0.03, 0.03, (half_r, half_c)) + 0.55

        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")
            
        self.current_height_field = height
        max_allowed_height = 0.6
        return np.clip(height, None, max_allowed_height)
    
    def inject_terrain_into_model(self, mj_model: mujoco.MjModel, 
                                   height_data: Optional[np.ndarray] = None) -> mujoco.MjModel:
        """将高度场数据注入 MuJoCo 模型"""
        if height_data is None:
            if self.current_height_field is None:
                height_data = self.generate_terrain()
            else:
                height_data = self.current_height_field
                
        if height_data.shape != (self.nrow, self.ncol):
            raise ValueError(f"Height data shape {height_data.shape} != ({self.nrow}, {self.ncol})")
            
        # 归一化到 [0, 1]
        h_min, h_max = height_data.min(), height_data.max()
        if h_max - h_min < 1e-8:
            h_max = h_min + 0.01
        normalized = (height_data - h_min) / (h_max - h_min)
        
        # 写入 hfield
        if mj_model.nhfield == 0:
            raise RuntimeError("Model has no hfield defined in XML!")
            
        mj_model.hfield_data[:] = normalized.ravel()
        
        # 修正：hfield_size 只有 4 列
        # [0,0]=x半长, [0,1]=y半长, [0,2]=最大高度, [0,3]=最小高度
        mj_model.hfield_size[0, 2] = h_max
        mj_model.hfield_size[0, 3] = h_min
        
        print(f"  [Terrain] Injected {self.terrain_type} terrain: height range [{h_min:.3f}, {h_max:.3f}]")
        
        return mj_model