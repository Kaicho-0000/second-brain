---
tags:
  - ML
  - learning-theory
  - generalization
  - AI
created: "2026-04-19"
status: draft
---

# 統計的学習理論

## 1. はじめに

統計的学習理論は、「なぜ機械学習は未知のデータに対してもうまく機能するのか?」という根本的な問いに答える理論的枠組みである。経験リスク最小化（ERM）の原理から始め、汎化誤差の解析、一様大数の法則まで体系的に学ぶ。

```mermaid
graph TD
    A[統計的学習理論] --> B[学習問題の定式化]
    B --> C[真のリスク R]
    B --> D[経験リスク R_n]
    C --> E[汎化誤差]
    D --> F[経験リスク最小化 ERM]
    E --> G[推定誤差]
    E --> H[近似誤差]
    F --> I[一様大数の法則]
    I --> J[汎化境界]
    J --> K[VC次元]
    J --> L[ラデマッハ複雑度]
    G --> M[バイアス-バリアンス分解]
```

## 2. 学習問題の定式化

### 2.1 基本設定

- **入力空間**: $\mathcal{X} \subseteq \mathbb{R}^d$
- **出力空間**: $\mathcal{Y}$（分類: $\{0, 1\}$ or $\{1, \ldots, K\}$、回帰: $\mathbb{R}$）
- **データ分布**: $\mathcal{D}$ on $\mathcal{X} \times \mathcal{Y}$（未知）
- **訓練データ**: $S = \{(x_1, y_1), \ldots, (x_n, y_n)\} \sim \mathcal{D}^n$ (i.i.d.)
- **仮説クラス**: $\mathcal{H} = \{h: \mathcal{X} \to \mathcal{Y}\}$
- **損失関数**: $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$

### 2.2 真のリスク（Population Risk）

$$R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(h(x), y)]$$

### 2.3 経験リスク（Empirical Risk）

$$\hat{R}_n(h) = \frac{1}{n}\sum_{i=1}^{n} \ell(h(x_i), y_i)$$

### 2.4 最適仮説

- ベイズ最適仮説: $h^* = \arg\min_{h: \mathcal{X} \to \mathcal{Y}} R(h)$
- 仮説クラス内最適: $h_{\mathcal{H}}^* = \arg\min_{h \in \mathcal{H}} R(h)$
- ERM: $\hat{h}_n = \arg\min_{h \in \mathcal{H}} \hat{R}_n(h)$

```python
import numpy as np

# 学習理論の基本概念のデモンストレーション
np.random.seed(42)

# 真のデータ生成過程
def true_function(x):
    return np.sin(2 * np.pi * x)

def generate_data(n, noise_std=0.3):
    x = np.random.uniform(0, 1, n)
    y = true_function(x) + noise_std * np.random.randn(n)
    return x, y

# 仮説クラス: k次多項式
def polynomial_features(x, degree):
    return np.column_stack([x**i for i in range(degree + 1)])

def fit_polynomial(x_train, y_train, degree):
    X = polynomial_features(x_train, degree)
    w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w

def predict(x, w):
    degree = len(w) - 1
    X = polynomial_features(x, degree)
    return X @ w

# 真のリスクを近似（大量のテストデータ）
x_test, y_test = generate_data(10000)

# 経験リスク vs 真のリスク
n_train = 30
x_train, y_train = generate_data(n_train)

print(f"{'次数':>4} | {'経験リスク':>10} | {'真のリスク':>10} | {'汎化ギャップ':>12}")
print("-" * 50)
for degree in [1, 3, 5, 9, 15]:
    w = fit_polynomial(x_train, y_train, degree)
    
    emp_risk = np.mean((predict(x_train, w) - y_train)**2)
    true_risk = np.mean((predict(x_test, w) - y_test)**2)
    gap = true_risk - emp_risk
    
    print(f"{degree:>4d} | {emp_risk:>10.4f} | {true_risk:>10.4f} | {gap:>12.4f}")
```

## 3. 経験リスク最小化（ERM）

### 3.1 ERM の原理

$$\hat{h}_n = \arg\min_{h \in \mathcal{H}} \hat{R}_n(h)$$

大数の法則により、$\hat{R}_n(h) \to R(h)$ as $n \to \infty$（各 $h$ について）。

しかし問題は: $\hat{h}_n$ は $\hat{R}_n$ に基づいて選ばれるため、$\hat{R}_n(\hat{h}_n) \leq R(\hat{h}_n)$（楽観的バイアス）。

### 3.2 汎化誤差の分解

$$R(\hat{h}_n) - R(h^*) = \underbrace{R(h_{\mathcal{H}}^*) - R(h^*)}_{\text{近似誤差}} + \underbrace{R(\hat{h}_n) - R(h_{\mathcal{H}}^*)}_{\text{推定誤差}}$$

- **近似誤差**: 仮説クラスの表現力の限界（$\mathcal{H}$ が大きいほど小さい）
- **推定誤差**: 有限サンプルによる不確実性（$\mathcal{H}$ が大きいほど大きい）

```mermaid
graph LR
    subgraph "汎化誤差の分解"
        A["R(ĥ_n) - R(h*)"] --> B["近似誤差<br/>R(h*_H) - R(h*)<br/><br/>仮説クラスの制約"]
        A --> C["推定誤差<br/>R(ĥ_n) - R(h*_H)<br/><br/>有限データの影響"]
    end
    B --> D["H を大きくすると減少"]
    C --> E["H を大きくすると増加<br/>n を大きくすると減少"]
    D --> F["トレードオフ"]
    E --> F
```

```python
import numpy as np

# 近似誤差と推定誤差の可視化
np.random.seed(42)

def experiment(n_train, degree, n_repeat=100):
    """多数の実験で推定誤差と近似誤差を分離"""
    x_test, y_test = generate_data(10000, noise_std=0.3)
    
    risks = []
    for _ in range(n_repeat):
        x_train, y_train = generate_data(n_train)
        w = fit_polynomial(x_train, y_train, degree)
        risk = np.mean((predict(x_test, w) - y_test)**2)
        risks.append(risk)
    
    return np.mean(risks), np.std(risks)

# ベイズリスク（ノイズの分散）
bayes_risk = 0.3**2

print("モデル複雑度と汎化誤差の関係 (n=30):")
print(f"{'次数':>4} | {'平均リスク':>10} | {'標準偏差':>10} | {'超過リスク':>10}")
print("-" * 50)
for degree in [1, 2, 3, 5, 9, 15, 25]:
    mean_risk, std_risk = experiment(30, degree, n_repeat=200)
    excess = mean_risk - bayes_risk
    print(f"{degree:>4d} | {mean_risk:>10.4f} | {std_risk:>10.4f} | {excess:>10.4f}")
```

## 4. 一様大数の法則

### 4.1 なぜ一様収束が必要か

各 $h$ について $\hat{R}_n(h) \to R(h)$ だけでは不十分。ERM は $\hat{R}_n$ を最小化する $h$ を選ぶため、全 $h \in \mathcal{H}$ に対して一様に収束する必要がある。

### 4.2 一様収束

$\mathcal{H}$ が一様収束するとは：

$$\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| \xrightarrow{P} 0 \quad (n \to \infty)$$

### 4.3 有限仮説クラスの場合

$|\mathcal{H}| < \infty$ のとき、union bound と Hoeffding の不等式から:

$$P\left(\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| > \epsilon\right) \leq 2|\mathcal{H}| \exp(-2n\epsilon^2)$$

$$\Rightarrow \text{確率 } 1 - \delta \text{ で: } \sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| \leq \sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}$$

```python
import numpy as np

# 一様収束の実験的検証
np.random.seed(42)

def uniform_convergence_experiment(n_hypotheses, n_samples, n_experiments=1000):
    """
    有限仮説クラスの一様収束を実験的に確認
    """
    # 真の分布: ベルヌーイ分布
    true_probs = np.random.uniform(0.2, 0.8, n_hypotheses)
    
    max_deviations = []
    for _ in range(n_experiments):
        # n_samples 個のデータでリスクを推定
        empirical_risks = np.zeros(n_hypotheses)
        for h in range(n_hypotheses):
            data = np.random.binomial(1, true_probs[h], n_samples)
            empirical_risks[h] = np.mean(data)
        
        max_dev = np.max(np.abs(empirical_risks - true_probs))
        max_deviations.append(max_dev)
    
    return np.array(max_deviations)

print("一様収束: sup |R(h) - R_n(h)| の分布")
for H_size in [10, 100, 1000]:
    for n in [50, 200, 1000]:
        deviations = uniform_convergence_experiment(H_size, n)
        mean_dev = np.mean(deviations)
        
        # 理論的上界
        delta = 0.05
        bound = np.sqrt(np.log(2 * H_size / delta) / (2 * n))
        
        # 経験的に95パーセンタイル
        empirical_95 = np.percentile(deviations, 95)
        
        print(f"  |H|={H_size:>4d}, n={n:>4d}: "
              f"平均={mean_dev:.4f}, 95%ile={empirical_95:.4f}, "
              f"理論上界={bound:.4f}")
```

## 5. ラデマッハ複雑度

### 5.1 定義

$$\hat{\mathfrak{R}}_n(\mathcal{H}) = \mathbb{E}_{\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^{n} \sigma_i h(x_i)\right]$$

$\sigma_i \in \{-1, +1\}$ はラデマッハ変数（等確率）。

### 5.2 汎化境界

確率 $1 - \delta$ で:

$$R(h) \leq \hat{R}_n(h) + 2\hat{\mathfrak{R}}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

```python
import numpy as np

def empirical_rademacher(H_predictions, n_rademacher=1000):
    """
    経験ラデマッハ複雑度を推定
    H_predictions: (n_hypotheses, n_samples) の予測値行列
    """
    n_hypotheses, n_samples = H_predictions.shape
    
    max_correlations = []
    for _ in range(n_rademacher):
        sigma = np.random.choice([-1, 1], n_samples)
        correlations = H_predictions @ sigma / n_samples
        max_correlations.append(np.max(correlations))
    
    return np.mean(max_correlations)

# 例: 線形分類器のラデマッハ複雑度
np.random.seed(42)
n_samples = 100
d = 10

X = np.random.randn(n_samples, d)

# 仮説クラス: w^T x (||w|| <= 1) の符号
n_hypotheses = 500
W = np.random.randn(n_hypotheses, d)
W = W / np.linalg.norm(W, axis=1, keepdims=True)

predictions = np.sign(W @ X.T)  # (n_hypotheses, n_samples)

rc = empirical_rademacher(predictions)
print(f"線形分類器のラデマッハ複雑度 (d={d}, n={n_samples}): {rc:.4f}")

# 次元が増えると複雑度も増加
for d in [2, 5, 10, 20, 50]:
    X = np.random.randn(n_samples, d)
    W = np.random.randn(n_hypotheses, d)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    predictions = np.sign(W @ X.T)
    rc = empirical_rademacher(predictions)
    print(f"  d={d:>2d}: Rademacher = {rc:.4f}")
```

## 6. ハンズオン演習

### 演習1: 汎化ギャップの観測

```python
import numpy as np

def exercise_generalization_gap():
    """
    異なるデータサイズ・モデル複雑度で汎化ギャップを観測せよ。
    """
    np.random.seed(42)
    
    print(f"{'n':>5} | {'degree':>6} | {'Train':>8} | {'Test':>8} | {'Gap':>8}")
    print("-" * 45)
    
    for n in [20, 50, 100, 500]:
        for degree in [1, 3, 7, 15]:
            train_risks = []
            test_risks = []
            
            for _ in range(100):
                x_train, y_train = generate_data(n)
                x_test, y_test = generate_data(1000)
                
                try:
                    w = fit_polynomial(x_train, y_train, min(degree, n-1))
                    tr = np.mean((predict(x_train, w) - y_train)**2)
                    te = np.mean((predict(x_test, w) - y_test)**2)
                    if te < 100:  # 発散を除外
                        train_risks.append(tr)
                        test_risks.append(te)
                except:
                    pass
            
            if train_risks:
                mean_tr = np.mean(train_risks)
                mean_te = np.mean(test_risks)
                gap = mean_te - mean_tr
                print(f"{n:>5d} | {degree:>6d} | {mean_tr:>8.4f} | "
                      f"{mean_te:>8.4f} | {gap:>8.4f}")

exercise_generalization_gap()
```

### 演習2: ERM の収束速度

```python
import numpy as np

def exercise_erm_convergence():
    """
    データサイズを増やした時の ERM 推定量の収束を確認せよ。
    """
    np.random.seed(42)
    
    degree = 5
    n_repeat = 200
    
    print("ERM の収束:")
    for n in [10, 20, 50, 100, 200, 500, 1000]:
        excess_risks = []
        for _ in range(n_repeat):
            x_train, y_train = generate_data(n)
            x_test, y_test = generate_data(5000)
            
            w = fit_polynomial(x_train, y_train, degree)
            test_risk = np.mean((predict(x_test, w) - y_test)**2)
            excess_risks.append(test_risk - 0.09)  # ベイズリスク近似
        
        mean_excess = np.mean(excess_risks)
        std_excess = np.std(excess_risks)
        print(f"  n={n:>4d}: E[excess risk]={mean_excess:.4f} "
              f"(±{std_excess:.4f})")
    
    print("\n→ 超過リスクは O(1/n) で減少（理論と一致）")

exercise_erm_convergence()
```

## 7. まとめ

| 概念 | 意味 | 実践での意義 |
|------|------|-------------|
| 経験リスク | 訓練データでの損失 | 訓練損失 |
| 真のリスク | 未知データでの期待損失 | テスト損失の期待値 |
| 汎化ギャップ | 真のリスク - 経験リスク | 過学習の度合い |
| 一様収束 | 全仮説で同時にリスクが収束 | ERM の正当化 |
| ラデマッハ複雑度 | 仮説クラスのデータ適合能力 | 汎化境界の鍵 |

## 参考文献

- Shalev-Shwartz, S. & Ben-David, S. "Understanding Machine Learning"
- Mohri, M. et al. "Foundations of Machine Learning"
- Vapnik, V. "The Nature of Statistical Learning Theory"
