class State:
    """ Represents state. """

    def __init__(self, rcp, rmsd):
        self.rcp = rcp
        self.rmsd = rmsd

    def __hash__(self):
        return hash((self.rcp, self.rmsd))

    def __eq__(self, value):
        if isinstance(value, State):
            return value.rcp == self.rcp and value.rmsd == self.rmsd
        return False

    def __repr__(self):
        return "<State rcp={} rmsd={}>".format(self.rcp, self.rmsd)
