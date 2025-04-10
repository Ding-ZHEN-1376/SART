import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
import time
import numba.c

# 系统几何参数
pNum = 400  # 图像像素个数
dNum = 512 # 探测器单元个数
views = 360  # 投影角度个数
sod = 600  # 源到物体距离
sdd = 1000  # 源到探测器距离
odd = sdd - sod
dsize = 2  # 探测器单元尺寸
maxIter = 10 # 最大迭代次数
lambda_ = 1  # 松弛因子

# 初始坐标向量
da = 2 * np.pi / views  # 角度间隔
L = dNum * dsize / 2  # 探测器半长
R = sod * L / np.sqrt(L ** 2 + sdd ** 2)  # 视野圆半径
pSize = R * 2 / pNum  # 像素尺寸
dx = 0.5 * pSize

# 探测器坐标
detX = np.linspace(-L + dsize / 2, L + dsize / 2, dNum)
detY = np.zeros_like(detX) + sdd

# 图像坐标（一定注意下面的坐标格式）
temp = np.linspace(-R + pSize / 2, R - pSize / 2, pNum)
imgX, imgY = np.meshgrid(temp, temp,indexing='ij')

# 创建一个模拟物体（圆形）
object_img = np.zeros((pNum, pNum))
center_x = pNum // 2
center_y = pNum // 2
radius = 50
y, x = np.ogrid[:pNum, :pNum]
circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
object_img[circle] = 1
# object_img = shepp_logan_phantom()
plt.imshow(object_img, cmap='gray')
plt.show()

# 积分坐标
int_ = np.linspace(-R, R, 200)
# 每条射线的sin和cos值
sinPerDet = detX / np.sqrt(detX ** 2 + detY ** 2)
cosPerDet = detY / np.sqrt(detX ** 2 + detY ** 2)
# 算出射线驱动的取样点在世界坐标系下的坐标
intX = sinPerDet[:, np.newaxis] * (int_ + sod)
intY = cosPerDet[:, np.newaxis] * (int_ + sod) - sod


# 正投影
# proj为投影矩阵，第k列用于存储投影数据
proj = np.zeros((dNum, views))
for k in range(views):
    theta = da * k
    rotx = np.cos(theta) * intX - np.sin(theta) * intY
    roty = np.sin(theta) * intX + np.cos(theta) * intY
    # 双线性插值求出每个点的投影
    interp_func = RegularGridInterpolator((temp, temp), object_img, method='linear',
                                          bounds_error=False, fill_value=0)
    nn=interp_func((rotx,roty))
    proj[:, k] = np.sum(nn, axis=1) * dx

plt.imshow(proj.T, cmap='gray')
plt.show()

# 反投影迭代重建
# 初始图像记为零矩阵
rec = np.zeros((pNum, pNum))
rec_1 = np.zeros((pNum, pNum))
# 用一个空列表去记录残差
residuals = []
temp_proj = np.zeros((dNum, views))
for iter_ in range(maxIter):
    start_time = time.time()
    IdxAng = np.random.permutation(views)
    for k in range(views):
        # 随机找一条射线
        theta = da * IdxAng[k]
        rotx = np.cos(theta) * intX - np.sin(theta) * intY
        roty = np.sin(theta) * intX + np.cos(theta) * intY
        # 双线性插值求出每个点的投影
        interp_func = RegularGridInterpolator((temp, temp), rec, method='linear',
                                              bounds_error=False, fill_value=0)
        nn = interp_func((rotx, roty))
        temp_proj[:, k] = np.sum(nn, axis=1) * dx
        # 计算残差
        delta = proj[:, IdxAng[k]] - temp_proj[:, k]
        delta /= pNum
        residuals.append(delta)
        # 用残差反投影更新图像
        rotx_bp = np.cos(theta) * imgX + np.sin(theta) * imgY
        roty_bp = -np.sin(theta) * imgX + np.cos(theta) * imgY
        uu = sdd * rotx_bp / (roty_bp + sod)
        f1 = interp1d(detX, delta, kind='linear', fill_value=0, bounds_error=False)
        deltaImg = f1(uu)
        rec += lambda_ * deltaImg
        rec[rec < 0] = 0

    if np.linalg.norm(rec - rec_1) < 1e-5:
        plt.imshow(rec, cmap='gray')
        plt.show()
        break
    else:
        rec_1 = rec.copy()
        plt.imshow(rec, cmap='gray')
        plt.show()


    end_time = time.time()
    print(f"迭代 {iter_ + 1} 耗时: {end_time - start_time} 秒")

