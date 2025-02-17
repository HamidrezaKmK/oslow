Given a permuation of the nodes $\boldsymbol{\sigma} = \left(\sigma_1, \ldots, \sigma_d \right)$ and some score vector $\boldsymbol{s} = \left(s_1, \ldots, s_d\right)$ for the nodes, we define the likelihood of the permutation under this score as

$$
\begin{aligned}
\log P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) =\log P_{Plackett-Luce} \left({\sigma_1}, \ldots, {\sigma_d}; \boldsymbol{s}\right) =  \sum_{j = 1}^d \left[\log s({\sigma_j}) - \log \left( \sum_{k=j}^d s({\sigma_k}) \right)\right]
\end{aligned}
$$

Given a dataset $\mathcal{D} = \left\{ \left(X^{(i)}_1, \ldots, X^{(i)}_d\right)\right\}_{i=1}^N$ and a flow model $f_\theta$, which takes a row of the data and a permutation as input and outputs the log-likelihood of observing that row under this permuation, we define the likelihood of $\mathcal{D}$ as follows:

$$
\begin{aligned}
\log P(\mathcal{D};\theta; \boldsymbol{s}) &= \sum_{i=1}^N \log P(X^{(i)}_1, \ldots, X^{(i)}_d;\theta; \boldsymbol{s})\\
 &= \sum_{i=1}^N \log \mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[P(X^{(i)}_1, \ldots, X^{(i)}_d;\theta; \boldsymbol{\sigma})\right]\\
 & =\sum_{i=1}^N \log \mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]
\end{aligned}
$$

Now, let's derive the maximum-likelihood estimation of parameters $\{\theta, \boldsymbol{s}\}$. We run a gradient-based optimization to find the best parameters:


$$
\begin{aligned}
\nabla_{\theta} \log P(\mathcal{D};\theta; \boldsymbol{s}) &= \sum_{i=1}^N \nabla_{\theta} \log P(X^{(i)}_1, \ldots, X^{(i)}_d;\theta; \boldsymbol{s})\\
 &= \sum_{i=1}^N \nabla_{\theta} \log \mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[P(X^{(i)}_1, \ldots, X^{(i)}_d;\theta; \boldsymbol{\sigma})\right]\\
 & =\sum_{i=1}^N \frac{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\nabla_{\theta} \exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]}{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[ \exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]} \\
  & =\sum_{i=1}^N \frac{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\nabla_{\theta} f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \cdot \exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]}{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[ \exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]}
\end{aligned}
$$

$$
\begin{aligned}
\nabla_{\boldsymbol{s}}\log P(\mathcal{D};\theta; \boldsymbol{s}) &= \sum_{i=1}^N \nabla_{\boldsymbol{s}} \log P(X^{(i)}_1, \ldots, X^{(i)}_d;\theta; \boldsymbol{s})\\
 &= \sum_{i=1}^N \nabla_{\boldsymbol{s}}\log \mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]\\
  &= \sum_{i=1}^N \frac{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\nabla_{\boldsymbol{s}} \log P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) \cdot \exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]}{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]}
\end{aligned}
$$


where,

$$
\begin{aligned}
&\nabla_{\boldsymbol{s}}\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]\\
 & = \nabla_{\boldsymbol{s}}\sum_{\boldsymbol{\sigma}} P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) \cdot \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right] \\
& = \sum_{\boldsymbol{\sigma}} \nabla_{\boldsymbol{s}} P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) \cdot \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right] \\
& = \sum_{\boldsymbol{\sigma}} P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) \frac{\nabla_{\boldsymbol{s}} P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s})}{P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s})} \cdot \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right] \\
& = \sum_{\boldsymbol{\sigma}} P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) \left(\nabla_{\boldsymbol{s}} \log P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s})\right) \cdot \left[\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right] \\
& =  \mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[\nabla_{\boldsymbol{s}} \log P_{Placket-Luce}(\boldsymbol{\sigma}; \boldsymbol{s}) \cdot \exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right]
\end{aligned}
$$

Therefore, if we define our loss function as following:
$$
\begin{aligned}
\ell(\theta, \boldsymbol{s}) = \frac{1}{N}\sum_{i=1}^N \frac{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[ \left(\log P_{Placket-Luce}(\boldsymbol{\sigma};\boldsymbol{s}) + f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right)\right) \cdot sg\left(\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right)\right]}{\mathbb{E}_{\boldsymbol{\sigma} \sim P_{Placket-Luce}(\cdot; \boldsymbol{s})} \left[ sg \left(\exp \left\{ f_\theta\left(X^{(i)}_1, \ldots, X^{(i)}_d; \boldsymbol{\sigma}\right) \right\}\right)\right]}
\end{aligned}
$$

where $sg$ means stop-gradient.
