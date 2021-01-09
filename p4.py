#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

#adapted from lab 9
def stability(A):
    '''
    Functions as a stability test for the result of the explicit FTCS scheme for solving Schroedinger's equation
    If rho > 1.0001, then the solution is unstable

    Parameters
    ----------
    A : matrix that is being tested for stability.

    Returns
    -------
    rho : the maximum eigenvalue of A, where rho > 1.0001 implies an unstable solution.
    '''
    
    eig = np.linalg.eig(A)[0]
    rho = np.absolute(np.max(eig))
    return rho

#adapted primarily from the schro function in NM4P
def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential = [], wparam = [10, 0, 0.5]):
    '''
    Solves the one-dimensional, time-dependent Schroedinger Equation using either the explicit FTCS scheme or the Crank-Nicholson scheme
    Calculates the conservation of probability throughout the computation and graphs it.
    
    Parameters
    ----------
    nspace : Number of spatial grid points
    ntime : Number of time steps to be evolved
    tau : Time step
    method : string, optional
             Scheme to be used ("ftcs" or "cn"). The default is 'ftcs'.
    length : float, optional
        Size of the spatial grid (extends from -length/2 to length/2). The default is 200.
    potential : 1-D array, optional
        The spatial indexes at which the potential should be set to 1. The default is [].
    wparam : list of initial parameters, optional
        Corresponds to [sigma0, x0, k0]. The default is [10, 0, 0.5].

    Returns
    -------
    allpsi : array_like
            the solutions to Schroedinger's equation at every time evaluated

    '''
    
    #* Initialize parameters 
    i_imag = 1j             # Imaginary i
    nspace = nspace #number of grid points
    length = length #llength of the system (extends from -length/2 to length/2)
    h = length/(nspace-1) 
    x = np.arange(nspace)*h - length/2.  # Coordinates  of grid points
    h_bar = 1.              # Natural units
    mass = 1/2.               # Natural units
    sigma0 = wparam[0]
    x0 = wparam[1]
    k0 = wparam[2]
    #* Initialize the wavefunction 
    velocity = 0.5   # Average velocity of the packet
    Norm_psi = 1./(np.sqrt(sigma0*np.sqrt(np.pi)))   # Normalization
    
    #Setting up the Hamiltonian operator matrix
    
    ham = np.zeros((nspace,nspace))     # Set all elements to zero
    coeff = -h_bar**2/(2*mass*h**2)
    for i in range(1,nspace-1) :
        ham[i,i-1] = coeff
        ham[i,i] = -2*coeff   # Set interior rows
        ham[i,i+1] = coeff
     
    #Setting up periodic boundary conditions by setting values for the first and last rows    
    ham[0,-1] = coeff;   ham[0,0] = -2*coeff;     ham[0,1] = coeff
    ham[-1,-2] = coeff;  ham[-1,-1] = -2*coeff;   ham[-1,0] = coeff
    
    #allows for user to input a potential
    for i in range(len(potential)):
        ham[potential[i], :] = ham[potential[i], :] + 1
        
    
    #Set up matrices corresponding to methods
    
    if (method == 'ftcs'):   
        #FTCS matrix
        matrix =np.identity(nspace)-0.5*i_imag*tau/h_bar*ham
        #Test stability
        if (stability(matrix)>1.0001): 
            print("The evolution is unstable!!!")
            return
        
    if (method == 'cn'):
        #Crank-Nicholson matrix
        matrix = np.dot(np.linalg.inv(np.identity(nspace) + 0.5*i_imag*tau/h_bar*ham),(np.identity(nspace)-0.5*i_imag*tau/h_bar*ham))
     
        
    psi = np.empty(nspace,dtype=complex)
    for i in range(nspace):
        psi[i]=Norm_psi*np.exp(i_imag*k0*x[i])*np.exp(-(x[i]-x0)**2/(2*sigma0**2))
    
    #Intialize loop and plot variables
    max_iter = int(length/(velocity*tau)+0.5)
    plot_iter=max_iter/8
    p_plot = np.empty((nspace,max_iter+1))
    p_plot[:,0]=np.absolute(psi[:])**2 #Record intial condition
    iplot = 0
    axisV = [-length/2.,length/2.,0.,max(p_plot[:,0])] #Fix axis min and max
    
    #Loop over desired number of steps
    allpsi = np.empty((ntime, nspace), dtype = complex)
    for iter in range (ntime):
        #New wave function with C-N
        psi=np.dot(matrix,psi)
        allpsi[iter:,] = np.dot(matrix, psi)
        #periodically record values for plotting
        if(iter+1)%plot_iter<1:
            iplot+=1
            p_plot[:,iplot] = np.absolute(psi[:])**2
            
       
    #Plot probability conservation        
    conservation = np.empty ((iplot+1))        
    for i in range(iplot+1): #Probability conservations
        conservation[i]=2*scint.trapz(p_plot[:,i])  
    plt.plot(range(iplot+1),conservation)  
    plt.title('Probability Conservation')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.show()
    plt.clf()
        
    return(allpsi)
    
def sch_plot(psi, t, plot_type, filename=None):
    '''
    Plots the solution of the Schroedinger equation at a specific time.
    Can plot either the real part of the wavefunction or the particle probability density.

    Parameters
    ----------
    psi : array_like
        Wavefunction, solution to the Schroedinger equation.
    t : integer
        Specific time at which the solution is to be visualized
    plot_type : string
        Type of plot ("psi" or "prob"), which correspond to a plot of the real part of the wavefunction
        or a plot of the particle probability density respectively.
    filename : string, optional
        Figures will be saved with the name given by the input. The default is None.

    Returns
    -------
    None.

    '''
    psi = psi[t,:]
    
    if plot_type == "psi":
        plt.clf()
    # plotting the real part of psi
        plt.plot(np.real(psi))
        plt.xlabel('position [x]')  
        plt.ylabel(r'$\psi(x)$')
        plt.title('Real Part of Initial Wave Function at Given Time')
        
        if filename==None:
            return
        else:
            plt.savefig(filename) 

    elif plot_type == "prob":
        plt.clf()
        plt.plot((psi)*np.conjugate(psi)) 
        plt.xlabel('position [x]')
        plt.ylabel('P(x,t)')
        plt.title('Probability Density at Given Time')
        plt.savefig(filename)
        
        if filename==None:
            return
        else:
            plt.savefig(filename) 
    
    else:
        print("Invalid plot type")
        return

                 
                 
                 