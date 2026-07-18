# Ergodic Control 学习与 PhD 申请计划

目标岗位：[PhD Student in Ergodic Control for Robotics](https://careers.werecruit.io/en/idiap/offres/phd-student-doctoral-researcher-in-ergodic-control-for-robotics-087a63)

目标不是只把 5 个 notebook 跑通，而是在最短时间内形成三项可以向面试官展示的证据：

1. 能从公式解释 ergodic metric、SMC、trajectory optimization 和 HEDAC；
2. 能独立复现实验、比较算法并解释失败模式；
3. 能提出一个与 DECOR 相关、可落到真实机器人或 UAV 的研究小项目。

岗位筛选已于 2026-07-15 开始，因此应当立即提交申请，同时用下面的项目成果持续增强材料，不要等全部学完再申请。

## 岗位与学习重点的对应关系

| 岗位/DECOR 需求 | 需要掌握的内容 | 可展示成果 |
|---|---|---|
| Optimal control、model-based optimization | ergodic metric、SMC、iLQR、约束与 receding horizon | SMC 与 iLQR 对比实验 |
| Diffusion-based ergodic coverage | HEDAC、热方程/Helmholtz 方程、Neumann 边界条件 | SMC 与 HEDAC 公平对比 |
| Curved geometry、anisotropic diffusion、graphs | Laplace–Beltrami、有限差分/有限元、扩散张量、图拉普拉斯 | 曲面或非凸区域 coverage demo |
| Small-data robot learning | 从 demonstration/uncertainty 构造目标分布，在线更新分布 | demonstration-conditioned target density |
| Real robots / Python components | 可复现环境、模块化实现、定量评估和可视化 | 干净代码、README、GIF、实验表格 |

## 第一阶段：14 天申请冲刺

建议每天投入 3–4 小时。每天都要留下可检查的 commit、公式笔记、图或实验结果。

### Day 1–2：ergodic metric

主线 notebook：`notebooks/ergodic_metric.ipynb`

需要能不看资料写出并解释：

\[
\phi_k=\int_{\mathcal X}p(x)f_k(x)\,dx,\qquad
c_k=\frac{1}{T}\int_0^T f_k(x(t))\,dt,
\]

\[
\mathcal E=\sum_k\lambda_k(c_k-\phi_k)^2,
\qquad
\lambda_k=(1+\lVert k\rVert)^{-(d+1)/2}.
\]

任务：

- 从 target density、trajectory empirical measure、Fourier coefficients 三步推导 metric；
- 改变 Fourier modes 数量，记录重建误差与计算量；
- 构造三条轨迹：均匀扫掠、从目标分布采样、只停在最高峰，比较 ergodic metric；
- 用一句话回答：为什么“访问最大概率点”不等于“匹配概率分布”？

交付物：`notes/01_ergodic_metric.md`、一张 metric 对比图、一个 2 分钟口头解释。

### Day 3–4：closed-form SMC feedback

主线 notebook：`notebooks/smc_ergodic_control.ipynb`

任务：

- 推导控制方向中历史覆盖误差与 \(\nabla f_k(x)\) 的作用；
- 解释 SMC 为什么是在线反馈，而不是预先生成固定 coverage path；
- 对比不同 mode 数、控制幅值、初始位置和非均匀 GMM；
- 添加边界处理，并记录控制振荡或覆盖不佳的场景。

交付物：`notes/02_smc.md`、参数消融表和一段轨迹 GIF。

### Day 5–7：ergodic trajectory optimization with iLQR

主线 notebook：`notebooks/ilqr_ergodic_control.ipynb`

任务：

- 复习 dynamics linearization、cost quadraticization、Riccati backward pass 和 forward rollout；
- 理解 ergodic cost 是整条轨迹上的非局部目标，以及 notebook 如何得到 \(a(t)=\partial \mathcal E/\partial x(t)\)；
- 分别跑 point-mass 与 differential-drive 示例；
- 加入 backtracking line search，并画 objective-versus-iteration；
- 比较 SMC 与 iLQR：feedback/开环、计算量、动力学约束、局部最优和在线重规划。

交付物：`notes/03_ilqr.md`、收敛曲线和一页算法比较表。

### Day 8–9：kernel metric 与 SO(3)

主线 notebook：

- `notebooks/kernel_ergodic_control.ipynb`
- `notebooks/kernel_ergodic_control_SO3.ipynb`

任务：

- 把 kernel ergodic metric 与 empirical-distribution discrepancy/MMD 联系起来；
- 解释它为何避免显式空间积分，以及何时比 Fourier basis 更容易扩展；
- 理解 kernel bandwidth 对局部/全局覆盖的影响；
- 跑通 JAX 版本，并理解 SO(3) 上的状态更新、log/exp map 和 manipulation relevance。

交付物：`notes/04_kernel_so3.md`、不同 bandwidth 的对比图。

这部分对岗位有价值，但优先级低于 HEDAC；时间不足时先完成二维 kernel demo，再进入下一阶段。

### Day 10–12：补齐岗位核心 HEDAC

参考实现：[ICRA 2024 HEDAC tutorial code](https://gitlab.com/sivic/icra2024-hedac)

重点论文与资料：

- [ICRA 2024 HEDAC tutorial slides](https://ergodiccontrol.github.io/pdfs/TutorialErgodicControl-ICRA2024-slides-deck05.pdf)
- [Whole-Body Ergodic Exploration with a Manipulator Using Diffusion](https://publications.idiap.ch/publications/show/5170)
- [Constrained Multi-Agent Ergodic Area Surveying with FEM](https://arxiv.org/abs/2109.10756)

需要掌握的计算链：

1. 用目标分布与累计覆盖构造 residual；
2. 将未覆盖区域编码成 source term；
3. 解 stationary/non-stationary diffusion 或 Helmholtz PDE；
4. 沿 potential field 的梯度生成控制；
5. 用边界条件、障碍物和扩散参数控制行为。

任务：

- 跑通 `hedac_basic`，画出 source、potential 和 vector field；
- 在同一个二维 target density、相同初始状态和时间预算下比较 SMC 与 HEDAC；
- 至少报告 coverage error、运行时间、边界/障碍物处理和参数敏感性；
- 能解释各向同性扩散与各向异性扩散的差别，以及扩散张量的物理意义。

交付物：`experiments/smc_vs_hedac/`，其中包含可复现实验脚本、README、图和结果表。

### Day 13–14：申请版 mini-project 与 research pitch

推荐选题：**Anisotropic ergodic coverage for robotic surface finishing**。

最小可行版本：

- 在二维或简单曲面上定义待喷涂/抛光强度分布；
- 实现 isotropic HEDAC baseline；
- 用位置相关扩散张量加入期望加工方向或曲面方向；
- 与 SMC/HEDAC baseline 比较 coverage error、路径长度、控制平滑度与计算时间；
- 输出 1 个 GIF、1 张总表和 2 页 research note。

两分钟 pitch 应回答：

1. 问题为什么重要；
2. 现有方法的具体限制；
3. 你的方法改了哪一个数学对象；
4. 如何验证，不只展示好看的轨迹；
5. 下一步怎样接真实机械臂或表面点云。

如果更想做 UAV，可把同一框架换成动态概率分布下的 multi-UAV search，并加入传感器 detection model 和分布在线更新。

## 第二阶段：4 周研究深化

### Week 3：PDE 与数值方法

- 推导 heat equation、Poisson/Helmholtz equation 与 Neumann boundary condition；
- 用有限差分实现规则网格 solver，验证网格分辨率与收敛误差；
- 学会 sparse matrix、conjugate gradient/multigrid 的基本使用；
- 在非凸域和障碍物场景测试 HEDAC。

### Week 4：曲面、图与各向异性扩散

- 学习 graph Laplacian 与 Laplace–Beltrami operator；
- 在 point cloud/mesh 上构造离散扩散；
- 实现 \(\nabla\cdot(D(x)\nabla u)\) 中的位置相关 SPD diffusion tensor；
- 比较 regular grid、mesh 与 graph 表示的复杂度和适用场景。

### Week 5：机器人系统化

- 把 coverage controller 与 robot dynamics/kinematics 分层；
- 加入 velocity、acceleration、collision 和 contact constraints；
- 用 receding horizon 或外环 coverage + 内环 tracking 组织系统；
- 设计仿真到真实机器人的日志、频率和安全接口。

### Week 6：研究输出

- 完成 3–5 页 technical report；
- 固化随机种子、配置、环境和一键复现命令；
- 做 5 页面试 slides：problem、metric、method、experiment、DECOR extension；
- 准备一个明确的失败案例和下一步研究问题。

## 阅读顺序

按“先建立共同语言，再贴近 DECOR”的顺序：

1. Mathew & Mezić, *Metrics for Ergodicity and Design of Ergodic Dynamics*；
2. Miller & Murphey, *Trajectory Optimization for Continuous Ergodic Exploration*；
3. Ivić et al., *Ergodicity-Based Cooperative Multiagent Area Coverage via a Potential Field*；
4. Bilaloglu et al., *Whole-Body Ergodic Exploration with a Manipulator Using Diffusion*；
5. Bilaloglu et al., [*Tactile Ergodic Coverage on Curved Surfaces*](https://publications.idiap.ch/publications/show/5541)；
6. Lanča et al., [*Ergodic Exploration of Dynamic Distribution*](https://arxiv.org/abs/2503.11235)；
7. Sun et al., *Fast Ergodic Search with Kernel Functions*；
8. Xu et al., [*Ergodic Imitation for Adaptive Exploration around Demonstrations*](https://arxiv.org/abs/2605.13996)。

每篇论文只做一页笔记：problem、state/control、target distribution、metric/PDE、solver、constraints、experiment、limitation、与 DECOR 的联系。

## 面试前验收标准

- [ ] 从零推导并解释 \(\phi_k\)、\(c_k\)、\(\lambda_k\) 和 ergodic metric；
- [ ] 不看代码讲清 SMC、iLQR、kernel metric、HEDAC 的信息流；
- [ ] 跑通仓库 5 个 notebook，并能修改 target/dynamics，而非只执行原单元格；
- [ ] 完成一次 SMC–HEDAC 公平对比；
- [ ] 解释 diffusion coefficient、source/sink、boundary condition 和 anisotropy；
- [ ] 展示一个与 surface finishing 或 UAV search 相关的 mini-project；
- [ ] 用定量指标讨论结果，并能解释至少一个失败案例；
- [ ] 准备 2 分钟、5 分钟和 15 分钟三个版本的项目介绍。

## Git 工作流

当前仓库约定：

- `origin`：个人 fork，用于提交学习笔记和实验；
- `upstream`：MurpheyLab 原仓库，用于同步教程更新。

同步上游时使用：

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

建议每个可验证成果单独提交，例如：

```text
notes: derive the spectral ergodic metric
experiment: compare SMC mode truncations
experiment: add SMC versus HEDAC benchmark
docs: add DECOR research pitch
```
