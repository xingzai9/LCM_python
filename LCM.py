import numpy as np
import matplotlib.pyplot as plt
import cv2
from pylab import mpl

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def LCM_computation(patch_LCM_in):
    row, col = patch_LCM_in.shape  # 对patch而言，行=列
    patch_LCM_in = np.array(patch_LCM_in, dtype=np.double)  # 改变数据类型
    # 分为3x3个cells，无论patch的尺寸，都是3*3个cell
    cell_size = row // 3

    # 计算中心cell的最大值 是一个标量
    L_n = np.max(patch_LCM_in[cell_size + 1:cell_size * 2, cell_size + 1:cell_size * 2])  # 选中patch中心区域，求最大值
    L_n_2 = L_n ** 2

    # 计算周边cell的均值,周边共3^2-1个cell,编号如下：
    # 1 2 3
    # 4 0 5
    # 6 7 8
    m_1 = np.mean(patch_LCM_in[0:cell_size, 0:cell_size])
    m_2 = np.mean(patch_LCM_in[0:cell_size, cell_size:cell_size * 2])
    m_3 = np.mean(patch_LCM_in[0:cell_size, cell_size * 2:cell_size * 3])
    m_4 = np.mean(patch_LCM_in[cell_size:cell_size * 2, 0:cell_size])
    m_5 = np.mean(patch_LCM_in[cell_size:cell_size * 2, cell_size * 2:cell_size * 3])
    m_6 = np.mean(patch_LCM_in[cell_size * 2:cell_size * 3, 0:cell_size])
    m_7 = np.mean(patch_LCM_in[cell_size * 2:cell_size * 3, cell_size:cell_size * 2])
    m_8 = np.mean(patch_LCM_in[cell_size * 2:cell_size * 3, cell_size * 2:cell_size * 3])

    # 计算C_n
    m_cell = np.array(
        [L_n_2 / m_1, L_n_2 / m_2, L_n_2 / m_3, L_n_2 / m_4, L_n_2 / m_5, L_n_2 / m_6, L_n_2 / m_7, L_n_2 / m_8])
    C_n = np.min(m_cell)

    # Replace the value of the central pixel with the Cn
    # patch_LCM_in[cell_size + 1:cell_size * 2, cell_size + 1:cell_size * 2] = C_n

    return C_n


def MLCM_computation(I_MLCM_in):
    I_MLCM_in = np.array(I_MLCM_in, dtype=np.double)
    row, col = I_MLCM_in.shape
    scales = np.array([9, 15, 21, 27])  # patch的尺寸有9x9,15x15,21x21,27x27
    l_max = scales.shape[0]  # 对应论文lmax=[1,4],l_max是4

    # Compute Cl according to Algorithm 1
    C_map_scales = np.zeros((row, col, l_max))
    for i in range(l_max):  # 对应不同尺度
        for j in range(0, row - scales[i] + 1):  # 单一尺度下以patch为单位做遍历，j是行
            for k in range(0, col - scales[i] + 1):  # k是列
                temp_patch = I_MLCM_in[j:j + scales[i], k:k + scales[i]]
                C_n = LCM_computation(temp_patch)  # 对patch执行Algorithm 1
                C_map_scales[j + scales[i] // 2, k + scales[i] // 2, i] = C_n

    # 这部分计算，生成4张对比度图，其中尺度最大的对比度图有效像元数最小，每个方向减去(scales(4)-1)/2=13
    max_margin = (scales[-1] - 1) // 2

    # 对4种尺度对比图的共同部分取最大值，作为输出
    C_hat = np.zeros((row - scales[-1] + 1, col - scales[-1] + 1))
    for i in range(row - scales[-1] + 1):
        for j in range(col - scales[-1] + 1):
            temp = np.array([
                C_map_scales[i + max_margin, j + max_margin, 0],
                C_map_scales[i + max_margin, j + max_margin, 1],
                C_map_scales[i + max_margin, j + max_margin, 2],
                C_map_scales[i + max_margin, j + max_margin, 3]
            ])
            C_hat[i, j] = np.max(temp)

    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # fig.suptitle('Contrast Maps at Different Scales')
    #
    # X, Y = np.meshgrid(np.arange(1, row + 1), np.arange(1, col + 1))
    #
    # for i, ax in enumerate(axs.flatten()):
    #     mesh = ax.pcolormesh(X, Y, C_map_scales[:, :, i], shading='auto', cmap='gray')
    #     ax.set_title(f'v={scales[i]}x{scales[i]} Contrast Map')
    #     ax.set_xlabel('row')
    #     ax.set_ylabel('col')
    #     fig.colorbar(mesh, ax=ax)
    #
    # plt.show()
    return C_hat, max_margin


def target_detection(C_hat, threshold, max_margin, I_in):
    # 用阈值生成mask
    row, col = C_hat.shape
    mask = np.zeros((row, col), dtype=np.uint8)
    target_pixel_num = 0  # 统计小目标在mask中占据的像元数

    for i in range(row):
        for j in range(col):
            if C_hat[i, j] > threshold:
                mask[i, j] = 1
                target_pixel_num += 1

    # 再把mask填入原图的中区域，四周各空max_margin
    row, col = I_in.shape
    I_out = np.zeros((row, col), dtype=np.uint8)
    I_out[max_margin:row - max_margin, max_margin:col - max_margin] = mask

    return I_out, target_pixel_num


def sqrt_matrix(C_hat, mean_C_hat):
    C_hat = np.array(C_hat, dtype=np.double)
    return np.var(C_hat, ddof=1)  # 使用ddof=1来得到样本方差


if __name__ == "__main__":
    # 读取测试图像
    I_read = cv2.imread("./unsuccess.png", cv2.IMREAD_GRAYSCALE)
    # 图像大小转换为 256*256
    I_read = cv2.resize(I_read, (256, 256), interpolation=cv2.INTER_NEAREST)
    # 检测RGB，转为灰度图
    if len(I_read.shape) == 3:  # 如果图像是3通道的，即RGB图像
        I_in = cv2.cvtColor(I_read, cv2.COLOR_BGR2GRAY)
    else:  # 如果图像已经是单通道的，即灰度图像
        I_in = I_read
    # 转为double型
    I_in = np.double(I_in)
    # 计算最终的Multiscale LCM
    [C_hat, max_margin] = MLCM_computation(I_in)

    # 计算均值、标准差
    mean_C_hat = np.mean(C_hat)
    sqrt_C_hat = np.sqrt(sqrt_matrix(C_hat, mean_C_hat))

    # 计算阈值
    k_Th = 4
    threshold = mean_C_hat + k_Th * sqrt_C_hat

    # 根据阈值判断，输出二值探测结果和统计小目标在mask中占据的像元数
    [I_out, target_pixel_num] = target_detection(C_hat, threshold, max_margin, I_in)
    print(target_pixel_num)
    # 显示输入和输出
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(I_in, cmap='gray', vmin=0, vmax=255)
    plt.title('原图')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(I_out, cmap='gray', vmin=0, vmax=1)
    plt.title('二值化输出')
    plt.axis('off')

    plt.show()
