class UpdateMethod:
    """
    UpdateMethod Superclass that defines the methods used for performing state updates
    """

    def __init__(self) -> None:
        """
        __init__ [summary]
        """
        pass

    def _calculate_neighbors(self):
        """
        _calculate_neighbors returns sparse matrix with protonation site distances
        """
        pass

    def _change_protonation_state(self, idx: int, new_charge: int):
        """
        _change_protonation_state changes the protonation state at atom idx

        Parameters
        ----------
        idx : int
            [description]
        new_charge : int
        """
        pass
    
    


class NaiveMCUpdate(UpdateMethod):
    """
    NaiveMCUpdate Performs naive MC update on molecule pairs in close proximity

    Parameters
    ----------
    UpdateMethod : [type]
        [description]
    """

    def __init__(self) -> None:
        pass


class StateUpdate:
    def __init__(self, system, updateMethod) -> None:
        self.system = system
        self.updateMethod = updateMethod

    def update(self):
        """
        updates the current state using the method defined in the UpdateMethod class
        """
        self.updateMethod.update(self.system)