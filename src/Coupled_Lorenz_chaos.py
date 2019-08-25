# cording: utf-8

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Coupled_Lorenz_Sys():

    def __init__(self, N=1):
        # number of Lorenz Oscillator
        self.N_osc = N
        self.set_init_state(np.array([[5, -9, 21] for i in range(0, self.N_osc) ]))
        self.set_const()

    # set the constant
    def set_const(self, omega_cnst = 10, r_cnst=28, b_cnst =8/3 , C_cnst = 0.01):
        #constant must be numpy array or flout.

        #self.omega is numpy array (size = N_osc)
        if(type(omega_cnst) is float):
            self.omega =  np.ones(self.N_osc) * omega_cnst
        elif(type(omega_cnst) is int):
            self.omega =  np.ones(self.N_osc) * float(omega_cnst)
        elif(len(omega_cnst) == self.N_osc):
            self.omega = omega_cnst
        else:
            self.omega = np.random.choice(omega_cnst, self.N_osc)

        
        #self.r is numpy array (size = N_osc)
        if(type(r_cnst) is float):
            self.r =  np.ones(self.N_osc) * r_cnst
        elif(type(r_cnst) is int):
            self.r =  np.ones(self.N_osc) * float(r_cnst)
        elif(len(r_cnst) == self.N_osc):
            self.r = r_cnst
        else:
            self.r = np.random.choice(r_cnst, self.N_osc)
            
        # self.b is numpy array (size = N_osc)
        if(type(b_cnst) is float):
            self.b =  np.ones(self.N_osc) * b_cnst
        elif(type(b_cnst) is int):
            self.b =  np.ones(self.N_osc) * float(b_cnst)
        elif(len(r_cnst) == self.N_osc):
            self.b = b_cnst
        else:
            self.b = np.random.choice(b_cnst, self.N_osc)

        # self.C is numpy matrix (size = (N_osc, N_osc))
        if(type(C_cnst) is float or type(C_cnst) is int):
            self.C =  np.zeros((self.N_osc, self.N_osc) )
            #print(self.C.shape)
            for i  in range(self.N_osc):
                #print("i:", i)
                if(i >= 1):
                    self.C[i, i - 1] = float(C_cnst)
            else:
                pass
        elif(len(C_cnst == self.N_osc) and len(C_cnst.shape) == 2  ):
            self.C = C_cnst
        else:
            self.C = np.random.choice(C_cnst, self.N_osc - 1)
    
    def dxdt(self, X, Y, Z):
        new_X = self.omega * (Y - X) + np.dot(self.C, X)
        return new_X

    def dydt(self, X, Y, Z):
        new_Y = self.r * X -Y - X*Z
        return new_Y

    def dzdt(self, X, Y, Z):
        new_Z = X * Y - self.b * Z
        return new_Z

    def set_init_state(self, init_vec):
        if(init_vec.shape[0] == self.N_osc and init_vec.shape[-1] == 3):
            self.init_vec = init_vec
        else:
            print("shape of init_vec must be (N, 3)")
        

    def solve(self, dt, N_t):
        self.dt = dt
        # XYZs is numpy arr for keep date of simulation (N, 3, t)
        XYZs = np.empty((self.N_osc, 3, N_t))
        Ts = np.linspace(0, self.dt * N_t, N_t)
        XYZs[:, :, 0] = self.init_vec

        for ii in range(N_t - 1):


            dX1 = self.dxdt(XYZs[:, 0, ii], XYZs[:, 1, ii], XYZs[:, 2, ii])
            dY1 = self.dydt(XYZs[:, 0, ii], XYZs[:, 1, ii], XYZs[:, 2, ii])
            dZ1 = self.dzdt(XYZs[:, 0, ii], XYZs[:, 1, ii], XYZs[:, 2, ii])

            dX2 = self.dxdt(XYZs[:, 0, ii] + 0.5*dX1*self.dt, XYZs[:, 1, ii] + 0.5*dY1*self.dt, XYZs[:, 2, ii] + 0.5*dZ1*self.dt)
            dY2 = self.dydt(XYZs[:, 0, ii] + 0.5*dX1*self.dt, XYZs[:, 1, ii] + 0.5*dY1*self.dt, XYZs[:, 2, ii] + 0.5*dZ1*self.dt)
            dZ2 = self.dzdt(XYZs[:, 0, ii] + 0.5*dX1*self.dt, XYZs[:, 1, ii] + 0.5*dY1*self.dt, XYZs[:, 2, ii] + 0.5*dZ1*self.dt)        

            dX3 = self.dxdt(XYZs[:, 0, ii] + 0.5*dX2*self.dt, XYZs[:, 1, ii] + 0.5*dY2*self.dt, XYZs[:, 2, ii] + 0.5*dZ2*self.dt)
            dY3 = self.dydt(XYZs[:, 0, ii] + 0.5*dX2*self.dt, XYZs[:, 1, ii] + 0.5*dY2*self.dt, XYZs[:, 2, ii] + 0.5*dZ2*self.dt)
            dZ3 = self.dzdt(XYZs[:, 0, ii] + 0.5*dX2*self.dt, XYZs[:, 1, ii] + 0.5*dY2*self.dt, XYZs[:, 2, ii] + 0.5*dZ2*self.dt)

            dX4 = self.dxdt(XYZs[:, 0, ii] + dX3*self.dt, XYZs[:, 1, ii] + dY3*self.dt, XYZs[:, 2, ii] + dZ3*self.dt)
            dY4 = self.dydt(XYZs[:, 0, ii] + dX3*self.dt, XYZs[:, 1, ii] + dY3*self.dt, XYZs[:, 2, ii] + dZ3*self.dt)
            dZ4 = self.dzdt(XYZs[:, 0, ii] + dX3*self.dt, XYZs[:, 1, ii] + dY3*self.dt, XYZs[:, 2, ii] + dZ3*self.dt)

            XYZs[:, 0, ii+ 1] =  XYZs[:, 0, ii] + (dX1 + 2.0 * dX2 + 2.0 * dX3 + dX4)*(self.dt/6.0)
            XYZs[:, 1, ii+ 1] =  XYZs[:, 1, ii] + (dY1 + 2.0 * dY2 + 2.0 * dY3 + dY4)*(self.dt/6.0)
            XYZs[:, 2, ii+ 1] =  XYZs[:, 2, ii] + (dZ1 + 2.0 * dZ2 + 2.0 * dZ3 + dZ4)*(self.dt/6.0)

        return XYZs, Ts


def main():
    sys1 = Coupled_Lorenz_Sys(N = 20 )
    XYZs, Ts = sys1.solve(0.01, 5000)
    
    plt.plot(XYZs[0, 0,:], XYZs[1, 0,:])
    fig = plt.figure()
    Ax = fig.add_subplot(111, projection='3d')
    Ax.scatter(XYZs[0, 0,:], XYZs[0, 1,:], XYZs[0, 2,:], s=5,   c=Ts, cmap="winter")
    # 軸ラベル
    Ax.set_xlabel('x')
    Ax.set_ylabel('y')
    Ax.set_zlabel('z')
    #Ax.axis('equal')
    # 表示
    plt.show()

if __name__ == '__main__':
    main()

