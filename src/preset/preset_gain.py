import numpy as np

from simtools.mpi_tools import join_data

class PresetGain:
    """
    
    Instance to initialize the Components Map-Making. It defines the input detectors gain.

    Self variables :    - gain_in: ndarray / if DB (Ndet, 2) / if UWB (Ndet)
                        - all_gain_in: ndarray / if DB (Ndet, 2) / if UWB (Ndet)
                        - gain_iter: ndarray / if DB (Ndet, 2) / if UWB (Ndet) 
                        - all_gain: ndarray / if DB (Ndet, 2) / if UWB (Ndet)
                        - all_gain_iter: ndarray / if DB (1, Ndet, 2) / if UWB (1, Ndet)

    """
    def __init__(self, preset_tools, preset_qubic):
        """
        Initialize the class with preset tools, qubic settings, and a seed value.

        Args:
            preset_tools: Class containing tools and simulation parameters.
            preset_qubic: Object containing qubic operator and variables.
        """
        ### Import preset QUBIC & tools
        self.preset_qubic = preset_qubic
        self.preset_tools = preset_tools

        ### Get input detectors gain
        self.preset_tools._print_message('    => Getting detectors gain')
        self._get_input_gain()

    def _get_input_gain(self):
        """
        Generates and processes input gain values for the instrument based on preset parameters.

        This method sets the `gain_in`, `all_gain_in`, `gain_iter`, `all_gain`, and `all_gain_iter`
        attributes of the instance. The gain values are generated using a normal distribution and may be
        adjusted based on the instrument type and preset parameters.

        Attributes:
            gain_in (numpy.ndarray): The generated gain values for the instrument.
            all_gain_in (numpy.ndarray): The combined gain input values across all processes.
            gain_iter (numpy.ndarray): The gain values adjusted for iteration, if fitting is enabled.
            all_gain (numpy.ndarray): The combined gain iteration values across all processes.
            all_gain_iter (numpy.ndarray): An array containing the gain iteration values.

        Raises:
            None
        """
        
        np.random.seed(None)
        if self.preset_qubic.params_qubic['instrument'] == 'UWB':
            self.gain_in = np.random.normal(1, self.preset_qubic.params_qubic['GAIN']['sig_gain'], self.preset_qubic.joint_in.qubic.ndets)
        else:
            self.gain_in = np.random.normal(1, self.preset_qubic.params_qubic['GAIN']['sig_gain'], (self.preset_qubic.joint_in.qubic.ndets, 2))

        self.all_gain_in = join_data(self.preset_tools.comm, self.gain_in)
        
        if self.preset_qubic.params_qubic['GAIN']['fit_gain']:
            gain_err = 0.2
            self.gain_iter = np.random.uniform(self.gain_in - gain_err/2, self.gain_in + gain_err/2, self.gain_in.shape)
        else:
            self.gain_iter = np.ones(self.gain_in.shape)
            
        self.all_gain = join_data(self.preset_tools.comm, self.gain_iter)
        self.all_gain_iter = np.array([self.gain_iter])