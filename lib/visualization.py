# __all__ = ['setup_seed', 'samples_plot', 'cov_square_exp', 'cov_exp', 'PCA', 'w1_dist_for_empirical',\
#            'SPDE_visual', 'std_cal']

# import numpy as np
# import random
# import scipy.linalg as la
# import scipy as sp
# import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
# import ot


# # function of setting random seed
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# # ----------------------------------------- GP case ----------------------------------------

# # function of ploting the sample path
# def samples_plot(data, flag_sensor, flag_lengthscale):
#     plt.fig, ax = plt.subplots()
#     x = np.linspace(-1,1,len(data[0,:]))
#     sensor_position = np.linspace(-1,1,flag_sensor)
#     lower_bound = np.min(np.min(np.array(data)))
#     upper_bound = np.max(np.max(np.array(data)))
#     for i in range(400):
#         ax.plot(x,data[i,:])
#         for k in range(flag_sensor):
#             plt.vlines(sensor_position[k], lower_bound, upper_bound, colors = "k", linestyles = "dashed")
#     # plt.savefig('results/Effect_of_sensor_number/sample_path_l={lengthscale}_sensor={num_sensor}.jpg'.format(lengthscale=flag_lengthscale, num_sensor=flag_sensor))
#     plt.show()

# # function for calculating the ground true covariance of square exponential kernel
# def cov_square_exp(xs, ys, l, sigma=1):
#     # Pairwise difference matrix.
#     dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
#     return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)

# # function for calculating the ground true covariance of exponential kernel
# def cov_exp(xs, ys, l, sigma=1):
#     # Pairwise difference matrix.
#     dx = np.abs(np.expand_dims(xs, 1) - np.expand_dims(ys, 0))
#     return (sigma ** 2) * np.exp(-(dx / l))

# # function to calculate the spectral of the samples of a distribution
# def PCA(data):
#     mean_val = np.mean(data, axis=0)
#     data_normalize = (data - mean_val)
#     cov = (1/len(data))*(data_normalize.T @ data_normalize)
#     a, b = la.eig(cov)
#     return a

# # function of checking the distance of the distribution
# def w1_dist_for_empirical(data1,data2):
#     data1 = torch.tensor(data1)
#     data2 = torch.tensor(data2)
#     n = len(data1)
#     data_dist1 = torch.zeros((n,1))
#     data_dist2 = torch.zeros((n,1))
#     for index in range(n):
#         data_dist1[index,0] = torch.norm(data1[index,:])
#         data_dist2[index,0] = torch.norm(data2[index,:])
#     C1 = sp.spatial.distance.cdist(data_dist1, data_dist1)       # Compute distance between each pair of the two collections of inputs
#     C2 = sp.spatial.distance.cdist(data_dist2, data_dist2)

#     p = ot.unif(n)                             # return a uniform histogram of length n_samples
#     q = ot.unif(n)                             # return a uniform histogram of length n_samples
#     gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=False, log=True)
    
#     return log0['gw_dist']

# # function of calculate std on the validation points
# def std_cal(A):
#     mean = torch.mean(A,axis=0)
#     A = (A - mean)
#     std = 0
#     for i in range(A.size(0)):
#         std += torch.norm(A[i,:])**2
#     std = torch.sqrt(std / A.size(0))
#     return std

# # ----------------------------------------- SDE case ----------------------------------------


# # ----------------------------------------- SPDE case ----------------------------------------

# def SPDE_visual(samples, figsize, root=None):
#     'description'
#     # this function is used to plot the mean value and the std of each point all over the whole domain
#     'input'
#     # samples: (nD) n-dimension array with the first dimension is batch size
#     'output'
#     # mean: (1D) mean of the samples
#     # std: (1D) std of the samples

#     # reshape the samples
#     std = []
#     new_samples = []
#     for i in range(np.size(samples, axis=0)):
#         point = np.reshape(samples[i],(1,-1),order='F')
#         new_samples.append(point)
#     samples = np.array(new_samples).squeeze(1)

#     num = np.size(samples, axis=1)
#     dim = int(np.sqrt(num))
#     # calculate the mean 
#     mean = np.mean(samples, axis=0)
#     std = np.std(samples, axis=0)
#     mean_plot = np.reshape(mean, (dim,dim), order='F')
#     std_plot = np.reshape(std, (dim,dim), order='F')

#     # plot the results
#     plt.figure(figsize = figsize)
#     ax1 = plt.subplot(1,2,1)
#     ax2 = plt.subplot(1,2,2)
#     plt.sca(ax1)
#     low = np.min(mean); high = np.max(mean)
#     sns.set()
#     sns.heatmap(mean_plot, vmin=low, vmax=high)   
#     plt.sca(ax2)
#     low = np.min(std); high = np.max(std)
#     sns.set()
#     sns.heatmap(std_plot, vmin=low, vmax=high)  
#     if root != None:
#         plt.savefig(root)
#     if root == None:
#         plt.show()

#     return mean, std









import sys
import os
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
import glob

# 忽略无关警告
warnings.filterwarnings("ignore")

# 1. 自动处理路径，确保能导入 lib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 模型所在的文件夹路径
model_dir = os.path.join(current_dir, 'trained_model/SDE')

def combine_xz(x, z):
    """辅助函数：适配 PIVAE_SDE 的输入拼接逻辑"""
    x_new = x.view(-1, 1)
    z_new = z.repeat(x.size(1), 1)
    return torch.cat((x_new, z_new), 1)

def main():
    # ==========================================
    # 🔍 智能查找模型文件
    # ==========================================
    if not os.path.exists(model_dir):
        print(f"错误：找不到文件夹 {model_dir}")
        print("请确认你是否运行过 SDE.py 并成功保存了模型？")
        return

    # 查找该目录下所有的 .pkl 文件
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))

    if not pkl_files:
        print(f"错误：在 {model_dir} 下没找到任何 .pkl 文件！")
        return

    # 默认使用找到的第一个文件 (通常这就是你刚才训练的)
    model_path = pkl_files[0]
    print(f"--- 自动找到模型文件 ---")
    print(f"路径: {model_path}")
    print(f"------------------------")

    # ==========================================
    # 🚀 加载与绘图逻辑
    # ==========================================
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 自动脱掉 DataParallel 马甲
    if hasattr(model, 'module'):
        model = model.module

    print("正在提取数据...")
    samples_u = []
    
    with torch.no_grad():
        z_batch = torch.randn(100, 4) 
        
        # 场景 A: 标准 VAE
        if hasattr(model, 'decoder'):
            print(">>> 检测到标准 VAE 结构 (decoder)")
            full_output = model.decoder(z_batch)
            samples_u = full_output[:, :400].numpy()
            
        # 场景 B: PI-VAE SDE 结构
        elif hasattr(model, 'u_decoder'):
            print(">>> 检测到 PI-VAE SDE 结构 (u_decoder)")
            u_coor = torch.linspace(-1, 1, 400).view(1, -1)
            for i in range(100):
                z_single = z_batch[i].view(1, -1)
                if hasattr(model, 'combine_xz'):
                    input_combined = model.combine_xz(u_coor, z_single)
                else:
                    input_combined = combine_xz(u_coor, z_single)
                u_pred = model.u_decoder(input_combined)
                samples_u.append(u_pred.view(-1).numpy())
            samples_u = np.array(samples_u)

        else:
            print(">>> 尝试使用 forward 推理...")
            try:
                u_coor = torch.linspace(-1, 1, 400).view(1, -1)
                # 尝试传入 None 或 dummy 变量
                output = model(None, None, None, u_coor, u_coor, u_coor) 
                samples_u = output[0].detach().numpy() if isinstance(output, tuple) else output.detach().numpy()
            except:
                print("！！！错误：无法识别模型结构！！！")
                return

    # --- 画 1D 曲线图 ---
    print("正在生成曲线对比图...")
    if samples_u.shape[1] != 400:
        samples_u = samples_u[:, :400]

    u_mean = np.mean(samples_u, axis=0)
    u_std = np.std(samples_u, axis=0)
    x_axis = np.linspace(-1, 1, 400)

    plt.figure(figsize=(10, 6))
    for i in range(10): 
        plt.plot(x_axis, samples_u[i], color='blue', alpha=0.1)
        
    plt.plot(x_axis, u_mean, color='red', linewidth=2, label='Mean Prediction')
    plt.fill_between(x_axis, u_mean - 2*u_std, u_mean + 2*u_std, color='red', alpha=0.2, label='Uncertainty')
    
    plt.title(f"Model Prediction\nSDE Solution u(x)", fontsize=14)
    plt.xlabel("Domain (x)")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    timestamp = int(time.time())
    save_path = os.path.join(current_dir, f'results/SDE/final_curve_{timestamp}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    print(f"\n✅ 成功！请打开新生成的图片: {save_path}")

if __name__ == '__main__':
    main()