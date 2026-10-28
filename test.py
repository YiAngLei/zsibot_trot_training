import jax
import jax.numpy as jp
from src.cfg.xg_config import get_xg_config  # 修改这里
from src.components.subcomponents.reset import ResetSubComponent

# 获取配置
config = get_xg_config()  # 修改这里
config.terrain.mesh_type = 'stairs'  # 确保是 mixed

# 打印确认配置
print(f"Terrain type: {config.terrain.mesh_type}")
print(f"Number of envs: {config.env.num_envs}")

# 模拟 ResetSubComponent
reset_comp = ResetSubComponent(
    model_nv=18,
    default_pose=jp.zeros(12),
    default_pose_mjx=jp.zeros(12),
    isaac_to_mjx_idx=jp.arange(12),
    mjx_to_isaac_idx=jp.arange(12),
    torque_his_num=3,
    use_custom_pd=False,
    kp_array=jp.zeros(12),
    kd_array=jp.zeros(12),
    torque_limits=jp.zeros(12),
    config=config,
    num_envs=config.env.num_envs,  # 使用配置中的环境数量
)

# 测试单个环境
rng = jax.random.PRNGKey(0)
result = reset_comp.build(rng=rng, randomize_dof_init=True)
print(f"单个环境 qpos: {result['qpos'][:3]}")  # 打印 x, y, z

# 使用 vmap 批量生成重置状态
@jax.vmap
def batch_reset(rng):
    return reset_comp.build(rng=rng, randomize_dof_init=True)

# 生成多个随机种子
num_test = 4096
rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, num_test)

# 批量生成
results = batch_reset(rngs)

# 统计位置分布
qpos = results['qpos']  # shape: (num_test, 18)
positions = qpos[:, :2]  # x, y

# 统计各象限数量
x = positions[:, 0]
y = positions[:, 1]

# 使用阈值判断
quadrant_flat = jp.sum((x >= -7.5) & (x <= -2.5) & (y >= -7.5) & (y <= -2.5))
quadrant_slope = jp.sum((x >= 1.5) & (x <= 6.5) & (y >= -7.5) & (y <= -2.5))
quadrant_stairs = jp.sum((x >= -7.5) & (x <= -2.5) & (y >= 1.5) & (y <= 6.5))
quadrant_rough = jp.sum((x >= 1.5) & (x <= 6.5) & (y >= 1.5) & (y <= 6.5))
other = num_test - (quadrant_flat + quadrant_slope + quadrant_stairs + quadrant_rough)

print("\n=== 地形分布统计 ===")
print(f"平地（左下）: {quadrant_flat} 个环境 ({quadrant_flat/num_test*100:.1f}%)")
print(f"斜坡（右下）: {quadrant_slope} 个环境 ({quadrant_slope/num_test*100:.1f}%)")
print(f"台阶（左上）: {quadrant_stairs} 个环境 ({quadrant_stairs/num_test*100:.1f}%)")
print(f"凹凸（右上）: {quadrant_rough} 个环境 ({quadrant_rough/num_test*100:.1f}%)")
print(f"其他区域: {other} 个环境 ({other/num_test*100:.1f}%)")
print(f"总计: {num_test}")

# 打印一些样本位置
print("\n=== 样本位置（前10个环境）===")
for i in range(min(10, num_test)):
    print(f"Env {i}: x={positions[i,0]:.3f}, y={positions[i,1]:.3f}")