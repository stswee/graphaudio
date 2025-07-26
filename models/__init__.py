from .gcn import TextGCN, MelGCN, MultiModalGCN
from .gat import TextGAT, MelGAT, MultiModalGAT
from .graphsage import TextGraphSAGE, MelGraphSAGE, MultiModalGraphSAGE

def get_model(model_name, **kwargs):
    """
    Factory to get model by name.
    model_name examples:
      - 'TextGCN', 'MelGAT', 'MultiModalGraphSAGE'
    kwargs:
      - input dims, mel shape, hidden dim, num_classes, etc.
    """
    models = {
        'TextGCN': TextGCN,
        'MelGCN': MelGCN,
        'MultiModalGCN': MultiModalGCN,
        'TextGAT': TextGAT,
        'MelGAT': MelGAT,
        'MultiModalGAT': MultiModalGAT,
        'TextGraphSAGE': TextGraphSAGE,
        'MelGraphSAGE': MelGraphSAGE,
        'MultiModalGraphSAGE': MultiModalGraphSAGE,
    }
    if model_name not in models:
        raise ValueError(f'Model {model_name} not found. Available: {list(models.keys())}')
    return models[model_name](**kwargs)
