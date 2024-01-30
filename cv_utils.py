import numpy as np
import matplotlib.pyplot as plt
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    np.random.seed(1338)
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm

    """为交叉验证对象的索引创建样本图."""
    # 为每个交叉验证分组生成训练/测试可视化图像
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):

         # 与训练/测试组一起填写索引
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # 可视化结果
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
        c=indices, marker='_', lw=lw, cmap=cmap_cv,
        vmin=-.2, vmax=1.2)

        # 将数据的分组情况和标签情况放入图像
        ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
            c=group, marker='_', lw=lw, cmap=cmap_data)

    # 调整格式
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
            xlabel='Sample index', ylabel="CV iteration",
            ylim=[n_splits+2.2, -.2], xlim=[0, 100])

    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)

    return ax
