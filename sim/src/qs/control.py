class SimulationControl:
    """Simple container for simulation completion flags."""
    def __init__(self):
        self.finished_bell = False
        self.finished_correction = False

    def stop_if_done(self):
        if self.finished_bell and self.finished_correction:
            import netsquid as ns
            ns.sim_stop()