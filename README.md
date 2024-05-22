# DYMAG

"Software package for investigation of graph expressivity"

## [`experiments`](./experiments)
Latest: for classification (and usage example), see [`PDEClassifier`](./experiments/classifier2.py#L143).  
for wrapped networks with batch normalization, use [`HeatBlock`](./experiments/classifier2.py#L15), [`WaveBlock`](./experiments/classifier2.py#L68), and [`LinearBlock`](./experiments/classifier2.py#L120) (with skip connection).  
use [`ChebyPolyLayer`](./experiments/cheby_poly_layer.py#L18) for a message passing class.  
Use [`get_cheby_coefs_heat`](./experiments/pde_layers.py#L96) and [`get_cheby_coefs_wave`](./experiments/pde_layers.py#L112) for computing the Chebyshev polynomial coefficients for the PDEs used by [`ChebyPolyLayer`](./experiments/cheby_poly_layer.py#L18).  
~For usage example, see this classifier that uses the `ChebyPolyLayer` [here](./experiments/classifier.py#L16).~ [DEPRECATED]
## [`src`](./src)
[`cross_validate.py`](./src/cross_validate.py) for cross validation  
[`prepare_dataset.py`](./src/prepare_dataset.py) for creating k-fold cross validation dataset, and also adding edge_weights as 1's (because they are missing from data). Now only works on data with existing node features.  
[`make_tuning_configs.py`](./src/make_tuning_configs.py) for creating the configuration yaml files and dSQ array job file for hyperparameter tuning.  
