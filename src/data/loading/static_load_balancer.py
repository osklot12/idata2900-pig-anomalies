from src.data.loading.load_balancer import LoadBalancer


class StaticLoadBalancer(LoadBalancer):
    """A load balancer that manages a static number of workers."""

    def run(self):
        pass

    def stop(self):
        pass
