# # import sys
# # import os
# # import torch
# # import numpy as np

# # # 强制将项目根目录加入搜索路径，解决 ModuleNotFoundError
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

# # from lib.visualization import SPDE_visual

# # # 1. 定位模型文件
# # model_path = './trained_model/SDE/model_ODE_u=2_k=17_f=21_datasize=2000_z=4.pkl'

# # if not os.path.exists(model_path):
# #     print(f"错误：找不到模型文件 {model_path}")
# # else:
# #     # 2. 加载模型数据
# #     print("正在加载模型数据...")
# #     # 注意：根据 PI-VAE 存储习惯，模型文件可能包含预测的 u, k, f
# #     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
# #     # 模拟从模型中提取样本进行可视化（这里以随机采样展示功能，你可以替换为真实的 model(x) 输出）
# #     # 假设你的 mesh_size 是 400
# #     display_samples = np.random.randn(100, 400) 
    
# #     # 3. 调用你修改过的可视化函数
# #     print("正在生成结果图...")
# #     # 将第 29 行修改为：
# #     save_path = 'final_prediction.png'
    
# #     # figsize 通常建议为 (12, 5) 以展示对比图
# #     SPDE_visual(display_samples, figsize=(12, 5), root=save_path)
    
# #     print(f"成功！请检查文件夹：{save_path}")









# import sys
# import os
# import torch
# import numpy as np
# import warnings
# import matplotlib.pyplot as plt
# import time
# import glob

# # 忽略无关警告
# warnings.filterwarnings("ignore")

# # 1. 自动处理路径，确保能导入 lib
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# # 模型期望所在的文件夹路径
# target_model_dir = os.path.join(current_dir, 'trained_model/SDE')

# def combine_xz(x, z):
#     """辅助函数：适配 PIVAE_SDE 的输入拼接逻辑"""
#     x_new = x.view(-1, 1)
#     z_new = z.repeat(x.size(1), 1)
#     return torch.cat((x_new, z_new), 1)

# def main():
#     print(f"--- 正在运行诊断程序 ---")
#     print(f"当前脚本位置: {current_dir}")
#     print(f"期望模型路径: {target_model_dir}")

#     # ==========================================
#     # 🔍 第一步：检查文件夹是否存在
#     # ==========================================
#     if not os.path.exists(target_model_dir):
#         print(f"\n❌ 错误：文件夹不存在！")
#         print(f"路径: {target_model_dir}")
#         print("建议：请确保你已经运行了 'mkdir -p examples/trained_model/SDE'")
#         return

#     # ==========================================
#     # 🔍 第二步：检查文件夹是否为空
#     # ==========================================
#     files_in_dir = os.listdir(target_model_dir)
#     if not files_in_dir:
#         print(f"\n❌ 错误：文件夹是空的！")
#         print(f"路径: {target_model_dir}")
#         print("原因推测：")
#         print("  1. 你可能手动创建了文件夹，但还没有重新运行 SDE.py 进行训练。")
#         print("  2. 上次训练在保存模型前就报错退出了。")
#         print("\n💡 解决方法：")
#         print("  请回到 examples 目录，重新运行一次 SDE.py 命令 (哪怕只跑 epoch=1 也可以测试生成文件)。")
#         return

#     print(f"文件夹内发现文件: {files_in_dir}")

#     # ==========================================
#     # 🔍 第三步：查找 .pkl 模型文件
#     # ==========================================
#     pkl_files = glob.glob(os.path.join(target_model_dir, "*.pkl"))

#     if not pkl_files:
#         print(f"\n❌ 错误：文件夹里有文件，但没有 .pkl 模型文件。")
#         return

#     # 默认使用找到的第一个文件 (按时间倒序，优先取最新的)
#     pkl_files.sort(key=os.path.getmtime, reverse=True)
#     model_path = pkl_files[0]
    
#     print(f"\n✅ 自动锁定最新的模型文件: {os.path.basename(model_path)}")

#     # ==========================================
#     # 🚀 加载与绘图逻辑
#     # ==========================================
#     try:
#         model = torch.load(model_path, map_location='cpu', weights_only=False)
#         model.eval()
#     except Exception as e:
#         print(f"模型加载失败: {e}")
#         return

#     # 自动脱掉 DataParallel 马甲
#     if hasattr(model, 'module'):
#         model = model.module

#     print("正在提取数据...")
#     samples_u = []
    
#     with torch.no_grad():
#         z_batch = torch.randn(100, 4) 
        
#         # 场景 A: 标准 VAE
#         if hasattr(model, 'decoder'):
#             print(">>> 检测到标准 VAE 结构 (decoder)")
#             full_output = model.decoder(z_batch)
#             samples_u = full_output[:, :400].numpy()
            
#         # 场景 B: PI-VAE SDE 结构
#         elif hasattr(model, 'u_decoder'):
#             print(">>> 检测到 PI-VAE SDE 结构 (u_decoder)")
#             u_coor = torch.linspace(-1, 1, 400).view(1, -1)
#             for i in range(100):
#                 z_single = z_batch[i].view(1, -1)
#                 if hasattr(model, 'combine_xz'):
#                     input_combined = model.combine_xz(u_coor, z_single)
#                 else:
#                     input_combined = combine_xz(u_coor, z_single)
#                 u_pred = model.u_decoder(input_combined)
#                 samples_u.append(u_pred.view(-1).numpy())
#             samples_u = np.array(samples_u)

#         else:
#             print(">>> 尝试使用 forward 推理...")
#             try:
#                 u_coor = torch.linspace(-1, 1, 400).view(1, -1)
#                 # 尝试传入 None 或 dummy 变量
#                 output = model(None, None, None, u_coor, u_coor, u_coor) 
#                 samples_u = output[0].detach().numpy() if isinstance(output, tuple) else output.detach().numpy()
#             except:
#                 print("！！！错误：无法识别模型结构！！！")
#                 return

#     # --- 画 1D 曲线图 ---
#     print("正在生成曲线对比图...")
#     if samples_u.shape[1] != 400:
#         samples_u = samples_u[:, :400]

#     u_mean = np.mean(samples_u, axis=0)
#     u_std = np.std(samples_u, axis=0)
#     x_axis = np.linspace(-1, 1, 400)

#     plt.figure(figsize=(10, 6))
#     for i in range(10): 
#         plt.plot(x_axis, samples_u[i], color='blue', alpha=0.1)
        
#     plt.plot(x_axis, u_mean, color='red', linewidth=2, label='Mean Prediction')
#     plt.fill_between(x_axis, u_mean - 2*u_std, u_mean + 2*u_std, color='red', alpha=0.2, label='Uncertainty')
    
#     plt.title(f"Model Prediction\nSDE Solution u(x)", fontsize=14)
#     plt.xlabel("Domain (x)")
#     plt.ylabel("u(x)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     timestamp = int(time.time())
#     save_path = os.path.join(current_dir, f'results/SDE/final_curve_{timestamp}.png')
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     plt.savefig(save_path)
#     print(f"\n✅ 成功！请打开新生成的图片: {save_path}")

# if __name__ == '__main__':
#     main()




# import sys
# import os
# import torch
# import numpy as np
# import warnings
# import matplotlib.pyplot as plt
# import time
# import glob

# warnings.filterwarnings("ignore")

# # 1. 路径设置
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# # 2. 自动查找模型
# model_dir = os.path.join(current_dir, 'trained_model/SDE')
# # 兼容 parent_dir
# if not os.path.exists(model_dir):
#     model_dir = os.path.join(parent_dir, 'trained_model/SDE')

# def main():
#     if not os.path.exists(model_dir):
#         print("❌ 找不到模型文件夹，请确认路径。")
#         return

#     pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
#     if not pkl_files:
#         print("❌ 文件夹为空。")
#         return

#     # 选最新的模型
#     pkl_files.sort(key=os.path.getmtime, reverse=True)
#     model_path = pkl_files[0]
#     print(f"--- 加载模型: {os.path.basename(model_path)} ---")

#     try:
#         model = torch.load(model_path, map_location='cpu', weights_only=False)
#         model.eval()
#     except Exception as e:
#         print(f"加载失败: {e}")
#         return

#     if hasattr(model, 'module'): model = model.module

#     # ==========================================
#     # 3. 提取全量数据 (u, k, f)
#     # ==========================================
#     print("正在提取全量物理场 (u, k, f)...")
    
#     samples_u = []
#     samples_k = []
#     samples_f = []
    
#     with torch.no_grad():
#         z_batch = torch.randn(100, 4) # 生成 100 个随机样本
        
#         # --- 场景 A: 简单结构 (Decoder 直接输出 1200 维) ---
#         if hasattr(model, 'decoder'):
#             full_output = model.decoder(z_batch) # Shape: (100, 1200)
#             print(">>> 结构识别: 单一 Decoder (拼接输出)")
            
#             # 按论文逻辑切分数据
#             # 通常顺序是 [u, k, f] 或 [k, u, f]，这里默认按 [u, k, f] 切分
#             # 如果画出来看着奇怪，可以尝试交换顺序
#             samples_u = full_output[:, 0:400].numpy()
#             samples_k = full_output[:, 400:800].numpy()
#             samples_f = full_output[:, 800:1200].numpy()
            
#         # --- 场景 B: PI-VAE SDE 结构 (独立 Decoder) ---
#         elif hasattr(model, 'u_decoder') and hasattr(model, 'k_decoder'):
#             print(">>> 结构识别: PI-VAE SDE (独立 Decoders)")
#             # 这种结构通常在 forward 里计算 f，这里简化处理，只取 u 和 k
#             u_coor = torch.linspace(-1, 1, 400).view(1, -1)
#             # 辅助函数
#             def combine(x, z): return torch.cat((x.view(-1,1), z.repeat(x.size(1),1)), 1)
            
#             list_u, list_k = [], []
#             for i in range(100):
#                 z_single = z_batch[i].view(1, -1)
#                 # 生成 u
#                 in_u = combine(u_coor, z_single)
#                 list_u.append(model.u_decoder(in_u).view(-1).numpy())
#                 # 生成 k
#                 in_k = combine(u_coor, z_single)
#                 list_k.append(model.k_decoder(in_k).view(-1).numpy())
            
#             samples_u = np.array(list_u)
#             samples_k = np.array(list_k)
#             # f 通常由自动微分算出来，这里暂时用 k 近似占位，或者留空
#             samples_f = np.zeros_like(samples_u) 
            
#         else:
#             print("❌ 无法识别模型输出结构")
#             return

#     # ==========================================
#     # 4. 绘图：复现论文 Figure 15 & 19
#     # ==========================================
#     print("正在生成三联画...")
#     x_axis = np.linspace(-1, 1, 400)
    
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     # 定义绘图辅助函数
#     def plot_field(ax, data, title, color_mean, color_sample, y_label):
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#         # 画样本
#         for i in range(15):
#             ax.plot(x_axis, data[i], color=color_sample, alpha=0.15, linewidth=1)
#         # 画均值
#         ax.plot(x_axis, mean, color=color_mean, linewidth=2.5, label='Mean')
#         # 画不确定性
#         ax.fill_between(x_axis, mean - 2*std, mean + 2*std, color=color_mean, alpha=0.2, label='Uncertainty (2std)')
#         ax.set_title(title, fontsize=14, fontweight='bold')
#         ax.set_xlabel('x')
#         ax.set_ylabel(y_label)
#         ax.grid(True, alpha=0.3)
#         ax.legend()

#     # --- 子图 1: u(x) 位移 ---
#     # 对应论文 Fig 15 左
#     plot_field(axes[0], samples_u, r'$u(x)$: Displacement', 'firebrick', 'salmon', 'u')

#     # --- 子图 2: k(x) 系数 ---
#     # 对应论文 Fig 15 右 (k通常是正数)
#     plot_field(axes[1], samples_k, r'$k(x)$: Coefficient', 'darkgreen', 'limegreen', 'k')

#     # --- 子图 3: f(x) 受力 ---
#     # 对应论文 Fig 19
#     plot_field(axes[2], samples_f, r'$f(x)$: Forcing Term', 'navy', 'cornflowerblue', 'f')

#     plt.tight_layout()
    
#     timestamp = int(time.time())
#     save_path = os.path.join(current_dir, f'results/SDE/paper_full_reproduction_{timestamp}.png')
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     plt.savefig(save_path, dpi=150)
#     print(f"\n✅ 成功！完整复现图已保存: {save_path}")
#     print("  - 左图: 对应论文 Fig 15 (u)")
#     print("  - 中图: 对应论文 Fig 15 (k)")
#     print("  - 右图: 对应论文 Fig 19 (f)")




# if __name__ == '__main__':
#     main()




import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import warnings

# =================配置区域=================
# 忽略无关的警告信息
warnings.filterwarnings("ignore")
# 设置随机种子保证结果可复现
torch.manual_seed(1234)
np.random.seed(1234)
# =========================================

# 1. 路径设置 (确保能导入 lib)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 2. 自动查找模型
model_dir = os.path.join(current_dir, 'trained_model/SDE')
# 兼容性检查：如果当前目录没有，去上一级找
if not os.path.exists(model_dir):
    model_dir = os.path.join(parent_dir, 'trained_model/SDE')

def get_latest_model(path):
    if not os.path.exists(path):
        return None
    files = glob.glob(os.path.join(path, "*.pkl"))
    if not files:
        return None
    # 按修改时间排序，取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    print(f"--- 启动物理场可视化脚本 ---")
    
    # --- 加载模型 ---
    model_path = get_latest_model(model_dir)
    if model_path is None:
        print(f"❌ 错误：在 {model_dir} 下找不到 .pkl 模型文件！请先运行 SDE.py 训练模型。")
        return

    print(f"✅ 锁定模型: {os.path.basename(model_path)}")
    
    # 使用 CPU 进行推理（绘图不需要 GPU，且方便处理数据）
    device = torch.device('cpu')
    
    try:
        # 加载完整模型
        model = torch.load(model_path, map_location=device)
        model.eval()
        # 处理 DataParallel 包装
        if hasattr(model, 'module'):
            model = model.module
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # ==========================================
    # 3. 核心逻辑：物理场生成与自动微分
    # ==========================================
    print("🚀 正在计算物理场 (使用自动微分计算 f)...")
    
    # 采样设置
    num_samples = 100  # 平行宇宙数量
    mesh_size = 400    # 空间分辨率
    latent_dim = 4     # 潜在空间维度 (通常是4，根据你SDE.py的设置)
    
    # 准备坐标 x (必须开启梯度追踪！)
    # Shape: (1, 400)
    x_tensor = torch.linspace(-1, 1, mesh_size).view(1, -1).to(device)
    x_tensor.requires_grad_(True) 

    # 准备潜在变量 z
    z_batch = torch.randn(num_samples, latent_dim).to(device)

    # 辅助函数：拼接 x 和 z
    def combine_xz(x, z):
        # x: (1, 400), z: (1, 4) -> (400, 5)
        x_repeated = x.view(-1, 1)
        z_repeated = z.repeat(x.size(1), 1)
        return torch.cat((x_repeated, z_repeated), 1)

    # 存储结果
    samples_u = []
    samples_k = []
    samples_f = []

    # 开始循环采样
    for i in range(num_samples):
        # 取出一个 z
        z_single = z_batch[i].view(1, -1)
        
        # 构造输入 (400, 5)
        # 注意：这里必须重新构造计算图，确保 x_tensor 的梯度能传导
        if hasattr(model, 'combine_xz'):
            # 如果模型自带拼接函数
            model_input = model.combine_xz(x_tensor, z_single)
        else:
            # 手动拼接
            model_input = combine_xz(x_tensor, z_single)

        # ---------------------------------------------
        # 核心步骤 A: 预测 u 和 k
        # ---------------------------------------------
        # PIVAE_SDE 结构通常有独立的 decoder
        if hasattr(model, 'u_decoder') and hasattr(model, 'k_decoder'):
            u_pred = model.u_decoder(model_input).view(1, -1) # (1, 400)
            k_pred = model.k_decoder(model_input).view(1, -1) # (1, 400)
        else:
            print("❌ 模型结构不匹配：找不到 u_decoder 或 k_decoder")
            return

        # ---------------------------------------------
        # 核心步骤 B: 自动微分计算 f (物理方程)
        # 方程: f = -0.1 * d/dx [ k * du/dx ]
        # ---------------------------------------------
        
        # 1. 求 du/dx
        # create_graph=True 是必须的，因为后面还要对它求二阶导
        u_x = torch.autograd.grad(
            outputs=u_pred, 
            inputs=x_tensor,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        # 2. 计算括号内的项: k * du/dx
        inner_term = k_pred * u_x

        # 3. 对括号内的项求导: d/dx [inner_term]
        # 这一步不需要 create_graph=True 了，因为不需要三阶导
        inner_x = torch.autograd.grad(
            outputs=inner_term,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(inner_term),
            create_graph=False
        )[0]

        # 4. 乘以系数 -1/10
        f_pred = -0.1 * inner_x

        # ---------------------------------------------
        # 收集数据 (转为 numpy)
        # ---------------------------------------------
        samples_u.append(u_pred.detach().numpy().flatten())
        samples_k.append(k_pred.detach().numpy().flatten())
        samples_f.append(f_pred.detach().numpy().flatten())

        if (i + 1) % 20 == 0:
            print(f"   已处理 {i + 1}/{num_samples} 个样本...")

    # 转为数组
    samples_u = np.array(samples_u)
    samples_k = np.array(samples_k)
    samples_f = np.array(samples_f)

    # ==========================================
    # 4. 绘图：复现论文
    # ==========================================
    print("🎨 正在绘制图像...")
    
    x_axis = np.linspace(-1, 1, mesh_size)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 绘图通用函数
    def plot_with_uncertainty(ax, x, data, title, color_mean, color_shade, y_label):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # 画前 20 个样本路径 (淡色细线)
        for i in range(min(20, len(data))):
            ax.plot(x, data[i], color=color_shade, alpha=0.15, linewidth=0.8)
        
        # 画均值 (深色粗线)
        ax.plot(x, mean, color=color_mean, linewidth=2.5, label='Mean')
        
        # 画 2倍标准差不确定性 (阴影)
        ax.fill_between(x, mean - 2*std, mean + 2*std, color=color_mean, alpha=0.2, label='Uncertainty (2std)')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

    # --- 左图: u(x) ---
    plot_with_uncertainty(
        axes[0], x_axis, samples_u, 
        r'$u(x)$: Displacement', 'firebrick', 'salmon', 'u'
    )

    # --- 中图: k(x) ---
    plot_with_uncertainty(
        axes[1], x_axis, samples_k, 
        r'$k(x)$: Coefficient', 'darkgreen', 'limegreen', 'k'
    )

    # --- 右图: f(x) ---
    plot_with_uncertainty(
        axes[2], x_axis, samples_f, 
        r'$f(x)$: Forcing Term', 'navy', 'cornflowerblue', 'f'
    )

    plt.tight_layout()
    
    # 保存结果
    timestamp = int(time.time())
    save_path = os.path.join(current_dir, f'results/SDE/PHYSICS_RECONSTRUCTION_{timestamp}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=200)
    print(f"\n✅ 成功！结果已保存至:\n   {save_path}")
    print("   这次 f(x) 应该会有数值和波动了！")

if __name__ == '__main__':
    main()