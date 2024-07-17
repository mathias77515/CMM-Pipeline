class PresetExternal:
    """
    
    Instance to initialize the Components Map-Making. It defines the external data.
    
    Self variables :    - external_nus : list (Nplanck)

    """
    def __init__(self, preset_tools):
        """
        Initialize the class with preset tools.

        Args:
            preset_tools: class containing tools and simulation parameters.
        """
        ### Define Planck parameters variable
        self.params_external = preset_tools.params['PLANCK']

        ### External frequencies
        preset_tools._print_message('    => Computing Planck frequency bands')
        self.external_nus = self._get_external_nus()

    def _get_external_nus(self):
        """
        Method to create a Python list of external frequencies by reading the `params.yml` file.

        This method reads the `params.yml` file and checks for the presence of specific frequency
        values under the 'PLANCK' section. If a frequency is present, it is added to the external
        frequencies list.

        Returns:
            list: A list of external frequencies selected in the `params.yml` file.
        """
        ### List of all Planck's frequency bands
        allnus = [30, 44, 70, 100, 143, 217, 353]

        ### Build variable with selected frequency bands
        external = []
        for nu in allnus:
            if self.params_external[f'{nu:.0f}GHz']:
                external += [nu]

        return external

