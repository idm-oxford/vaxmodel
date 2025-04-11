import numpy as np
from scipy.integrate import solve_ivp

class TwoEpitopeModel:
    def __init__(self, params):
        """
        Initialise the antibody dynamics model.
        
        """
        self._params = params

    def _right_hand_side(self, t, y):
            """
            Constructs the RHS of the equations of the system of ODEs for given a
            time point.

            Parameters
            ----------
            t : float
                Time point at which we compute the evaluation.
            y : numpy.array
                Array of all the compartments of the ODE system.

            Returns
            -------
            numpy.array
                Matrix representation of the RHS of the ODEs system.

            """
            # Split compartments into their types

            # Free antigen, Complement-bound ICs
            H_XS, C_XS = y[0], y[1]

            # Antibody bound ICs
            H_OS, H_XO, H_OO = y[2], y[3], y[4]

            # B cells
            B_X, B_S = y[5], y[6]

            # Short-lived plasma cells
            S_X, S_S = y[7], y[8]

            # Long-lived plasma cells
            L_X, L_S = y[9], y[10]

            # Antibodies
            A_X, A_S = y[11], y[12]

            # Read parameters
            k = self._params['k']
            dAg = self._params['dAg']
            if self._params['ICE']==1:
                dIC = self._params['dIC']
            elif self._params['ICE']==0:
                dIC = dAg
            dAb = self._params['dAb']
            s = self._params['s']
            phi = self._params['phi']
            pp = self._params['pp']
            kslow = self._params['kslow']
            dB = self._params['dB']
            nn = self._params['nn']
            r = self._params['r']
            CC = self._params['CC']
            c = k*CC
            dS = self._params['dS']
            dL = self._params['dL']
            f = self._params['f']
            gamma_XS= self._params['IM'][0,1]
            gamma_SX= self._params['IM'][1,0]
            
            Ag_B_X  = C_XS + gamma_SX*H_XO
            Ag_B_S  = C_XS + gamma_XS*H_OS

            dydt = np.array((
                (r - dAg)*H_XS - c*H_XS - k*H_XS*(A_X + kslow*A_S),
                c*H_XS - k*C_XS*(A_X + kslow*A_S) - dAg*C_XS,
                k*((H_XS+C_XS)*A_X - H_OS*(gamma_XS*kslow*A_S)) - dIC*H_OS,
                k*((H_XS+C_XS)*kslow*A_S - H_XO*(gamma_SX*A_X)) - dIC*H_XO,
                k*(H_OS*gamma_XS*kslow*A_S + H_XO*gamma_SX*A_X) - dIC*H_OO,
                (0.5*s*B_X*Ag_B_X**nn)/((phi*self._params['PHI'][0])**nn + Ag_B_X**nn) - dB*B_X,
                (0.5*s*B_S*Ag_B_S**nn)/((phi*self._params['PHI'][1]/kslow)**nn + Ag_B_S**nn) - dB*B_S,
                (0.5-f)*s*B_X*Ag_B_X**nn/((phi*self._params['PHI'][0])**nn + Ag_B_X**nn) - dS*S_X,
                (0.5-f)*s*B_S*Ag_B_S**nn/((phi*self._params['PHI'][1]/kslow)**nn + Ag_B_S**nn) - dS*S_S,
                (f*s*B_X*Ag_B_X**nn)/((phi*self._params['PHI'][0])**nn + Ag_B_X**nn) - dL*L_X,
                (f*s*B_S*Ag_B_S**nn)/((phi*self._params['PHI'][1]/kslow)**nn + Ag_B_S**nn) - dL*L_S,
                pp*(S_X + L_X) - dAb*A_X,
                pp*(S_S + L_S) - dAb*A_S,
                ))

            return dydt

    def run(self, times):
            """
            Computes the values in each compartment of the ODE system.

            Parameters
            ----------
            times : list
                List of time points at which vaccination occurs.
            M : int
                Number of doses.

            Returns
            -------
            list
                Solution of the ODE system at the time points provided.

            """
            # Initial conditions
            pre_immunity = [self._params['B_0'], self._params['B_0'],
                            self._params['Ab_0'], self._params['Ab_0']]
            bx, bs, ax, as_ = pre_immunity

            if len(self._params['IC_adm']) == 0:
                IC_adm = [0, 0, 0, 0]
            else:
                IC_adm = self._params['IC_adm']

            pp = self._params['pp']
            d_Ab = self._params['d_Ab']
            lx = d_Ab*ax/pp
            ls = d_Ab*as_/pp

            init = [self._params['dose'][0], *IC_adm, bx, bs, 0, 0, lx, ls, ax, as_]

            y = []
            t = []

            # Solve the system of ODEs
            sol = solve_ivp(
                lambda t, y: self._right_hand_side(
                    t, y),
                [times[0], times[1]], init)
            
            y.append(sol.y[11]+sol.y[12])
            t.append(sol.t)

            # Boosts
            mm=2 # This is to deal with M=1
            if self._params['Het']==0:
                while(mm<=self._params['M']): # Does not execute boosts if M=1
                    init = sol.y[:,-1]
                    init[0] = init[0]+self._params['dose'][mm-1]
                    init[1:5] = [x + y for x, y in zip(init[1:5], IC_adm)]
                    sol = solve_ivp(
                            lambda t, y: self._right_hand_side(
                                t, y),
                [times[mm-1], times[mm]], init)
                    y.append(sol.y[11]+sol.y[12])
                    t.append(sol.t)
                    mm = mm+1
            return t, y
