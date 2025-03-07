from abc import ABC, abstractmethod

class LoadBalancer:
    """An interface for load balancers."""

    @abstractmethod
    def run(self):
        """Runs the load balancer."""

    @abstractmethod
    def stop(self):
        """Stops the load balancer."""