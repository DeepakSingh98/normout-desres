from robustbench.utils import load_model

def robustbench_model(model_name: str):
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    return model
