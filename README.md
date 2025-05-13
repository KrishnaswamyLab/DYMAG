# DYMAG

"Software package for investigation of graph expressivity"

## [`experiments`](./experiments)
Latest: for classification (and usage example), see [`PDEClassifier`](./experiments/classifier2.py#L143).  
for wrapped networks with batch normalization, use [`HeatBlock`](./experiments/classifier2.py#L15), [`WaveBlock`](./experiments/classifier2.py#L68), and [`LinearBlock`](./experiments/classifier2.py#L120) (with skip connection).  
use [`ChebyPolyLayer`](./experiments/cheby_poly_layer.py#L18) for a message passing class.  
Use [`get_cheby_coefs_heat`](./experiments/pde_layers.py#L96) and [`get_cheby_coefs_wave`](./experiments/pde_layers.py#L112) for computing the Chebyshev polynomial coefficients for the PDEs used by [`ChebyPolyLayer`](./experiments/cheby_poly_layer.py#L18).  
