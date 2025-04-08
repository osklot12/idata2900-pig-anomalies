from src.data.preprocessing.augmentation.plan.augmentation_plan import AugmentationPlan


class AugmentationPlanFactory:
    """Factory for augmentation plans."""

    def __enter__(self):
        self._current_plan = AugmentationPlan.random()
        return self

    def __exit__(self, *args):
        self._current_plan = None

    def get_current_plan(self) -> AugmentationPlan:
        """
        Returns the current AugmentationPlan.

        Returns:
            AugmentationPlan: current AugmentationPlan
        """
        if self._current_plan is None:
            raise RuntimeError("Augmentation plan not set")

        return self._current_plan