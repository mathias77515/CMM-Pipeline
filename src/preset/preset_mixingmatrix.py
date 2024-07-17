import numpy as np

import pysm3
import pysm3.units as u

class PresetMixingMatrix:
    """
    Instance to initialize the Components Map-Making. It defines the input Mixing Matrix.

    Self variables :    - nus_eff_in: ndarray (Nsub_in + Nintegr * Nplanck)
                        - nus_eff_out: ndarray (Nsub_out + Nintegr * Nplanck)
                        - Amm_in: ndarray (Nsub_in + Nintegr * Nplanck, Ncomp)
                        - beta_in: ndarray / if d1 (12*nside_beta_in**2, Ncomp-1) / if not (Ncomp-1)

    """
    def __init__(self, preset_tools, preset_qubic, preset_fg):
        """
        Initialize the class with preset tools, qubic, and foreground.

        Args:
            preset_tools: Class containing tools and simulation parameters.
            preset_qubic: Class containing qubic operator and variables.
            preset_fg: Class containing foregrounds variables.
        """
        # Import preset Foregrounds, QUBIC & tools
        self.preset_tools = preset_tools
        self.preset_qubic = preset_qubic
        self.preset_fg = preset_fg

        ### Get input spectral index
        self.preset_tools._print_message('    => Building Mixing Matrix')
        self._get_beta_input()

    def extra_sed(self, nus, correlation_length):
        """
        Calculates the extra SED (Spectral Energy Distribution) based on the given parameters.

        Args:
            self: The instance of the class.
            nus (array-like): The array of frequencies.
            correlation_length (float): The correlation length.

        Returns:
            array-like: The array of extra SED values.
        """

        np.random.seed(1)
        extra = np.ones(len(nus))

        if self.preset_fg.params_foregrounds['Dust']['model_d'] != 'd6':
            return np.ones(len(nus))
        
        else:
            for ii, i in enumerate(nus):
                rho_covar, rho_mean = pysm3.models.dust.get_decorrelation_matrix(353.00000001 * u.GHz, 
                                           np.array([i]) * u.GHz, 
                                           correlation_length=correlation_length*u.dimensionless_unscaled)
                rho_covar, rho_mean = np.array(rho_covar), np.array(rho_mean)
                extra[ii] = rho_mean[:, 0] + rho_covar @ np.random.randn(1)

            return extra

    def _get_Amm(self, comps, comp_name, nus, beta_d=None, beta_s=None, init=False):
        """
        Compute the mixing matrix A for given components and frequencies.

        Parameters:
        comps (list): List of component objects.
        comp_name (list): List of component names corresponding to `comps`.
        nus (list): List of frequency values.
        beta_d (float, optional): Spectral index for Dust component. Defaults to 1.54.
        beta_s (float, optional): Spectral index for Synchrotron component. Defaults to -3.
        init (bool, optional): Flag to indicate if this is an initialization step. Defaults to False.

        Returns:
        np.ndarray: The computed mixing matrix A of shape (len(nus), len(comps)).
        """

        # Set default spectral indices if not provided
        if beta_d is None:
            # beta_d = 1.54
            beta_d = np.random.normal(
                        self.preset_fg.params_foregrounds['Dust']['beta_d_init'][0], 
                        self.preset_fg.params_foregrounds['Dust']['beta_d_init'][1], 
                        1
                    )
        if beta_s is None:
            # beta_s = -3
            beta_s = np.random.normal(
                        self.preset_fg.params_foregrounds['Synchrotron']['beta_s_init'][0], 
                        self.preset_fg.params_foregrounds['Synchrotron']['beta_s_init'][1], 
                        1
                    )
            
        # Determine the number of components and frequencies
        ncomp, nfreq = len(comps), len(nus)
        # Initialize the mixing matrix with zeros
        A = np.zeros((nfreq, ncomp))
        
        # Check if the Dust model is 'd6' and not in initialization step
        if self.preset_fg.params_foregrounds['Dust']['model_d'] == 'd6' and init == False:
            # Compute extra scaling factor for Dust component
            extra = self.extra_sed(nus, self.preset_fg.params_foregrounds['Dust']['l_corr'])
        else:
            # Default scaling factor is 1 for all frequencies
            extra = np.ones(nfreq)

        for inu, nu in enumerate(nus):
            for jcomp in range(ncomp):
                # If the component is CMB, set the mixing matrix value to 1
                if comp_name[jcomp] == 'CMB':
                    A[inu, jcomp] = 1.
                # If the component is Dust, evaluate the component and apply the extra scaling factor
                elif comp_name[jcomp] == 'Dust':
                    A[inu, jcomp] = comps[jcomp].eval(nu, np.array([beta_d]))[0][0] * extra[inu]
                # If the component is Synchrotron, evaluate the component
                elif comp_name[jcomp] == 'Synchrotron':
                    A[inu, jcomp] = comps[jcomp].eval(nu, np.array([beta_s]))
        return A
    
    def _spectral_index_modifiedblackbody(self, nside):
        """
        Method to define input spectral indices if the d1 model is used for thermal Dust description.

        Parameters:
        nside (int): The nside parameter defines the resolution of the HEALPix map.

        Returns:
        numpy.ndarray: An array containing the spectral indices for the thermal Dust model.
        """
        sky = pysm3.Sky(nside=nside, preset_strings=['d1'])
        return np.array(sky.components[0].mbb_index)
    
    def _spectral_index_powerlaw(self, nside):
        """
        Define input spectral indices if the s1 model is used for Synchrotron description.

        Parameters:
        nside (int): The nside parameter for the Sky object.

        Returns:
        np.array: Array of spectral indices for the Synchrotron component.
        """
        sky = pysm3.Sky(nside=nside, preset_strings=['s1'])
        return np.array(sky.components[0].pl_index)

    def _get_beta_input(self):
        """
        Define the input spectral indices based on the model type.

        If the model is 'd0' or 'd6', the input spectral index is fixed (1.54).
        Otherwise, the model assumes varying spectral indices across the sky by calling the previous method.
        In this case, the shape of beta is (Nbeta, Ncomp).

        This method sets the following attributes:
        - self.nus_eff_in: Effective input frequencies as a numpy array.
        - self.nus_eff_out: Effective output frequencies as a numpy array.
        - self.Amm_in: Amplitude matrix for input components.
        - self.beta_in: Spectral indices for input components.

        Raises:
            TypeError: If the dust model is not implemented.
        """
        self.nus_eff_in = np.array(list(self.preset_qubic.joint_in.qubic.allnus) + list(self.preset_qubic.joint_in.external.allnus))
        self.nus_eff_out = np.array(list(self.preset_qubic.joint_out.qubic.allnus) + list(self.preset_qubic.joint_out.external.allnus))
        
        if self.preset_fg.params_foregrounds['Dust']['model_d'] in ['d0', 'd6']:
            self.Amm_in = self._get_Amm(self.preset_fg.components_model_in, self.preset_fg.components_name_in, self.nus_eff_in, init=False)
            self.Amm_in[len(self.preset_qubic.joint_in.qubic.allnus):] = self._get_Amm(self.preset_fg.components_model_in, self.preset_fg.components_name_in, self.nus_eff_in, init=True)[len(self.preset_qubic.joint_in.qubic.allnus):]
            if self.preset_fg.params_foregrounds['CO']['CO_in']:
                self.beta_in = np.array([float(i._REF_BETA) for i in self.preset_fg.components_model_in[1:-1]])
            else:
                self.beta_in = np.array([float(i._REF_BETA) for i in self.preset_fg.components_model_in[1:]])
            
        elif self.preset_fg.params_foregrounds['Dust']['model_d'] == 'd1':
            self.Amm_in = None
            self.beta_in = np.zeros((12*self.preset_fg.params_foregrounds['Dust']['nside_beta_in']**2, len(self.preset_fg.components_in)-1))
            for iname, name in enumerate(self.preset_fg.components_name_in):
                if name == 'CMB':
                    pass
                elif name == 'Dust':
                    self.beta_in[:, iname-1] = self._spectral_index_modifiedblackbody(self.preset_fg.params_foregrounds['Dust']['nside_beta_in'])
                elif name == 'Synchrotron':
                    self.beta_in[:, iname-1] = self._spectral_index_powerlaw(self.preset_fg.params_foregrounds['Dust']['nside_beta_in'])
        else:
            raise TypeError(f"{self.preset_fg.params_foregrounds['Dust']['model_d']} is not yet implemented...")
        
    def _get_index_seenpix_beta(self):
        """
        Method to initialize index seenpix beta variable
        """

        if self.preset.fg.params_foregrounds['fit_spectral_index']:
            self._index_seenpix_beta = 0
        else:
            self._index_seenpix_beta = None