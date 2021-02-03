#! / us r / bin /env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time

beg = time.time()


# Particle - has coordinates x, y, z as attributes.
# Default particle mass = 1.
class Particle:
    def __init__(self, x, y, z, mass=1):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass


# Point - has coordinates, assigned mass, and overdensity as attributes.
# Points form the *grid*, i.e. their positions are quantised. This is handled
# in the grid class. Particles on the other hand can, in principle, be anywhere.
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.mass = 0
        self.delta = 0


# A cell is defined by its boundaries, i.e. initial (x,y,z) and width;
# cells have a mass attribute to store total mass in cell.
class Cell:
    def __init__(self, index, x, y, z, width, mass=0):
        self.index = index
        self.x_low = x
        self.x_high = x+width
        self.y_low = y
        self.y_high = y+width
        self.z_low = z
        self.z_high = z+width
        self.mass = mass


# A grid with dim = dimension, defined cell_width. Init method initialises cells in
# the requested grid, as well as the grid points.
class Grid:
    def __init__(self, dimension, cell_width, cells=[], particles=[]):
        self.dimension = dimension
        self.cells = cells
        self.particles = particles
        self.cell_width = cell_width
        self.n_particles = len(self.particles)
        self.mean_density = 0

        self.initialise_cells()
        self.initialise_grid_points()
        self.grid_points = np.array(self.grid_points)

    def initialise_cells(self):
        ind = 0

        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    self.cells.append(Cell(ind, i+0.5, j+0.5, k+0.5, width=self.cell_width))
                    ind += 1

    def initialise_grid_points(self):
        self.grid_points = []

        # Not ideal to use 3 nested loops, but for such low dimension of
        # the problem this is not too bad.
        for i in range(int(self.dimension)):
            for j in range(int(self.dimension)):
                for k in range(int(self.dimension)):
                    self.grid_points.append(Point(i+0.5, j+0.5, k+0.5))

    # add particle at (x, y, z) to the grid:
    def add_particle(self, x, y, z, mass=1):
        self.particles.append(Particle(x, y, z, mass))
        self.n_particles = len(self.particles)

    # Cloud-in-cell method: assign masses to the grid points by weighing the mass of each
    # particle based on the distance to its nearest 8 grid points. Total sum of weights = 1.
    def assign_masses_cic(self):
        ngridpoints = len(self.grid_points)

        # Loop for all particles:
        for ind in range(self.n_particles):
            [x, y, z] = [self.particles[ind].x, self.particles[ind].y, self.particles[ind].z]

            for ind_point in range(ngridpoints):
                xp = self.grid_points[ind_point].x
                yp = self.grid_points[ind_point].y
                zp = self.grid_points[ind_point].z
                # Periodic boundary conditions:
                if abs(x-xp) <= self.cell_width or abs(x-xp-self.dimension) <= self.cell_width or abs(x-xp+self.dimension) <= self.cell_width:
                    if abs(y - yp) <= self.cell_width or abs(y-yp-self.dimension) <= self.cell_width or abs(y-yp+self.dimension) <= self.cell_width:
                        if abs(z - zp) <= self.cell_width or abs(z-zp-self.dimension) <= self.cell_width or abs(z-zp+self.dimension) <= self.cell_width:
                            # print('Closeby point: ', xp, yp, zp)
                            dx = min(abs(x-xp), abs(x-xp-self.dimension), abs(x-xp+self.dimension))
                            dy = min(abs(y-yp), abs(y-yp-self.dimension), abs(y-yp+self.dimension))
                            dz = min(abs(z-zp), abs(z-zp-self.dimension), abs(z-zp+self.dimension))
                            # print(dx, dy, dz)
                            self.grid_points[ind_point].mass += (1-dx)*(1-dy)*(1-dz)
        self.calculate_mean_density()
        self.calculate_density_contrast()

    # mean density = \bar(\rho) = total mass in particles divided by the grid volume
    def calculate_mean_density(self):
        self.mean_density = len(self.particles)/self.dimension**3

    # density contrast = \delta = (rho - \bar(rho))/\bar(rho):
    # calculate the local density at each grid point, then translate to delta
    def calculate_density_contrast(self):
        ngridpoints = len(self.grid_points)
        for i in range(ngridpoints):
            rho = self.grid_points[i].mass/self.cell_width**3
            self.grid_points[i].delta = (rho-self.mean_density)/self.mean_density


ndim = 3
nparticles = 1024
ngrid = 16

# Initialise the particle positions:
np.random.seed(121)
positions = np.random.uniform(low=0, high=16, size=(3, 1024))
x = positions[0]
y = positions[1]
z = positions[2]

# Intialise grid with cell width of unity:
grid = Grid(dimension=ngrid, cell_width=1)

# Add all particles to the grid, and assign masses to the grid points:
[grid.add_particle(x[i], y[i], z[i]) for i in range(len(x))]
grid.assign_masses_cic()
masses = [grid.grid_points[i].mass for i in range(16**3)]

print('Conservation of mass check:')
print('No. of particles of mass 1 = ', nparticles)
print('Total mass assigned to the grid points = ', sum(masses))


grid_x = np.array([grid.grid_points[i].x for i in range(16**3)])
grid_y = np.array([grid.grid_points[i].y for i in range(16**3)])
grid_z = np.array([grid.grid_points[i].z for i in range(16**3)])
grid_delta = np.array([grid.grid_points[i].delta for i in range(16**3)])

for z in [4.5, 9.5, 11.5, 14.5]:
    # find the grid points close to the desired slice: (of course, this is just 1 example way to do it)
    ind = np.isclose(grid_z, z, atol=0.05)
    plt.figure()
    plt.scatter(grid_x[ind], grid_y[ind], c=grid_delta[ind], marker='s', s=170)
    cbar = plt.colorbar()
    cbar.set_label('Density contrast $\delta$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grid slice at z='+str(z))
    plt.savefig('plots/grid-'+str(math.floor(z))+'.png', dpi=300)
    plt.close()


# Simple swap function:
def swap(a, b):
    temp = a
    a = b
    b = temp
    return a, b


# Shuffle the array x by reversing the indices of its elements (in binary)
def bit_reversal_shuffle(x):
    n = len(x)
    # nbits = int(math.log(n, 2))
    for i in range(n):
        i_bitwise = '{0:011b}'.format(i)    # some python magic - use format function to get binary representation
        i_bitwise_reversed = i_bitwise[::-1]
        j = int(i_bitwise_reversed, 2)      # use int function to convert binary -> decimal
        # print(i, '->', j)
        x[i], x[j] = swap(x[i], x[j])
    return x


# Discrete Fourier transform:
def dft(x):
    N = len(x)
    n = np.arange(N)
    k = np.arange(N)
    H = []
    for i in range(N):
        factor = np.exp(-2j*np.pi*k[i]*n/N)
        H.append(sum(x*factor))
    return H


# Discrete Fourier transform:
def idft(x):
    N = len(x)
    n = np.arange(N)
    k = np.arange(N)
    H = []
    for i in range(N):
        factor = np.exp(2j*np.pi*k[i]*n/N)
        H.append(sum(x*factor))
    return np.array(H).astype(complex)/2


# Supposedly working version of a Fast Fourier Transform:
def fft(x):
    x = x.astype(complex)
    N = len(x)
    a = np.zeros(N).astype(complex)  # to hold the FT during recursive steps
    k = np.arange(N)

    if N == 2:  # do DFT on 2-element pairs:
        # print('do dft')
        return dft(x)
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        w = np.exp(-2j*np.pi*k/N)

        # do butterfly 'swap'
        for i in range(len(even)):
            a[i] = even[i] + w[i]*odd[i]
            a[i+len(even)] = even[i] + w[i+len(even)]*odd[i]

        return a


# (?) working version of an Inverse Fast Fourier Transform:
def ifft(x):
    x = x.astype(complex)
    N = len(x)
    a = np.zeros(N).astype(complex)  # to hold the FT during recursive steps
    k = np.arange(N)

    if N == 2:  # do DFT on 2-element pairs:
        # print('do dft')
        return idft(x)
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        w = np.exp(2j*np.pi*k/N)

        # do butterfly 'swap'
        for i in range(len(even)):
            a[i] = even[i] + w[i]*odd[i]
            a[i+len(even)] = even[i] + w[i+len(even)]*odd[i]

        return a


# 3-dimensional FFT;
def fft_3d(x):
    n = len(x)
    ft = np.empty(x.shape).astype(complex)
    row, column, height = x.shape
    # loop over rows, columns, and then vertically
    for r in range(row):
        for c in range(column):
            ft[r, c, :] = fft(x[r, c, :])
    for c in range(column):
        for h in range(height):
            ft[:, c, h] = fft(ft[:, c, h])
    for r in range(row):
        for h in range(height):
            ft[r, :, h] = fft(ft[r, :, h])

    # ft = [fft_2d(x[i]) for i in range(n)]
    # ft = np.array(ft).astype(complex)
    return ft


# 3-dimensional IFFT;
def ifft_3d(x):
    n = len(x)
    ft = np.empty(x.shape).astype(complex)
    row, column, height = x.shape
    # loop over vertical dimension, rows, and then columns; order doesn't matter, though
    for r in range(row):
        for c in range(column):
            ft[r, c, :] = ifft(x[r, c, :])
    for c in range(column):
        for h in range(height):
            ft[:, c, h] = ifft(ft[:, c, h])
    for r in range(row):
        for h in range(height):
            ft[r, :, h] = ifft(ft[r, :, h])
    return ft/n     # divide by n - IFT convention


delta_ft = np.array(fft_3d(np.array(grid_delta).reshape((16, 16, 16))))
print('Check: does the FT implementation match the numpy one:')
print(np.allclose(delta_ft, np.fft.fftn(np.array(grid_delta).reshape((16, 16, 16)))))

# k-vector initialisation:
k_squared = np.zeros((16, 16, 16))
for i in range(16):
    for j in range(16):
        for k in range(16):
            k_squared[i, j, k] = i*i + j*j + k*k

k_squared[0, 0, 0] = 1  # k(0, 0, 0) is just normalisation, so we can set it to one

phi_ft = delta_ft/k_squared     # FT of potential
phi = np.array(ifft_3d(phi_ft))  # potential = IFT of FT of potential
print('Does the IFT match?')
print(np.allclose(phi, np.fft.ifftn(phi_ft)))

# Plot the slices:
for z in [4.5, 9.5, 11.5, 14.5]:
    slice = math.floor(z)
    plt.figure()
    plt.imshow(np.log10(abs(phi_ft[:, :, slice])))
    cbar = plt.colorbar()
    cbar.set_label('log$_{10}|\widetilde{\Phi}|$')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title('Grid slice at z='+str(z))
    plt.savefig('plots/grid-phi-ft-'+str(math.floor(z))+'.png', dpi=300)
    plt.close()

    plt.figure()
    plt.imshow(abs(phi[:, :, slice]))
    cbar = plt.colorbar()
    cbar.set_label('$|\Phi|$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grid slice at z='+str(z))
    plt.savefig('plots/grid-phi-'+str(math.floor(z))+'.png', dpi=300)
    plt.close()

end = time.time()
print('Ex. 2 took ', round(end-beg), 's')
