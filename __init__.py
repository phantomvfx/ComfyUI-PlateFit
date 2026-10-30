from .plate_nodes import FitToPlateNode, RelightTransferNode

NODE_CLASS_MAPPINGS = {
    "PlateFit_FitToPlate": FitToPlateNode,
    "PlateFit_RelightTransfer": RelightTransferNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlateFit_FitToPlate": "Fit to Plate (PlateFit)",
    "PlateFit_RelightTransfer": "Relight Transfer (PlateFit)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
