from robustbench.utils import load_model

def robustbench_model(model_name: str, get_robustbench_layers: bool):
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')

    if get_robustbench_layers:
        layers = []
        
        for module in model.children():
            layers.append(module)

        return layers
    
    else:
        return model
