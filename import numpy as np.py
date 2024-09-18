import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import completeness_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import HistGradientBoostingRegressor

# K-Means con matrices dispersas
rng = np.random.RandomState(0)
X_blobs, y_blobs = make_blobs(random_state=rng)
X_sparse = scipy.sparse.csr_matrix(X_blobs)
X_train_sparse, X_test_sparse, _, y_test_sparse = train_test_split(X_sparse, y_blobs, random_state=rng)
kmeans = KMeans(n_init="auto").fit(X_train_sparse)
print("K-Means Completeness Score:", completeness_score(kmeans.predict(X_test_sparse), y_test_sparse))

# Regressor con HistGradientBoostingRegressor y dependencias parciales
n_samples = 500
X = rng.randn(n_samples, 2)
noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
y = 5 * X[:, 0] + np.sin(10 * np.pi * X[:, 0]) - noise

gbdt_no_cst = HistGradientBoostingRegressor().fit(X, y)
gbdt_cst = HistGradientBoostingRegressor(monotonic_cst=[1, 0]).fit(X, y)

disp = PartialDependenceDisplay.from_estimator(gbdt_no_cst, X, features=[0], feature_names=["feature 0"],
                                               line_kw={"linewidth": 4, "label": "unconstrained", "color": "tab:blue"})
PartialDependenceDisplay.from_estimator(gbdt_cst, X, features=[0],
                                        line_kw={"linewidth": 4, "label": "constrained", "color": "tab:orange"},
                                        ax=disp.axes_)
disp.axes_[0, 0].plot(X[:, 0], y, "o", alpha=0.5, zorder=-1, label="samples", color="tab:green")
plt.legend()
plt.show()

# Lasso con soporte de pesos de muestra
n_samples, n_features = 1000, 20
X_reg, y_reg = make_regression(n_samples, n_features, random_state=rng)
sample_weight = rng.rand(n_samples)
X_train_reg, X_test_reg, y_train_reg, y_test_reg, sw_train, sw_test = train_test_split(X_reg, y_reg, sample_weight, random_state=rng)

lasso = Lasso()
lasso.fit(X_train_reg, y_train_reg, sample_weight=sw_train)
print("Lasso Score:", lasso.score(X_test_reg, y_test_reg, sample_weight=sw_test))
