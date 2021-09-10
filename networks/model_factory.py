import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, trainer):
        
        if dataset == 'CIFAR100':
            
            if trainer == 'wild' or trainer == 'wild_exp':
                import networks.wrn as res
                return res.wrn(100)
            
            import networks.resnet32 as res
            return res.resnet32(100, trainer)
        
        if dataset == 'CIFAR10':
            
            import networks.resnet18 as res
            return res.resnet32(10, trainer)
        
        elif dataset == 'Imagenet' or dataset == 'VggFace2_1K' or dataset == 'Google_Landmark_v2_1K':
            
            import networks.resnet18 as res
            return res.resnet18(1000, trainer)
        
        elif dataset == 'VggFace2_5K':
            
            import networks.resnet18 as res
            return res.resnet18(5000, trainer)
        
        elif dataset == 'Google_Landmark_v2_10K':
            
            import networks.resnet18 as res
            return res.resnet18(10000, trainer)
