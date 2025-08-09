# airbrakes-lookup-generator

Generator for lookup table to control Hyperion II airbrakes

Note, need to manually add one line to RocketPy's flight.py file to fix a bug with initiallizing the time on a simulation starting from an initially flying state. After line 1173, add:
            self.t_initial = self.initial_solution[0] # HOTFIX