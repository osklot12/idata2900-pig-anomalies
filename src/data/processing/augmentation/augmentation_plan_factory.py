from src.data.processing.augmentation.augmentation_plan import AugmentationPlan


class AugmentationPlanFactory:
    """Factory for augmentation plans."""

    def get_plan(self) -> AugmentationPlan:
        """
        Returns the current AugmentationPlan.

        Returns:
            AugmentationPlan: current AugmentationPlan
        """
        return AugmentationPlan.random()