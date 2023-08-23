#!/usr/bin/env python
# coding: utf-8

# # PyGeM
# ## Tutorial 5: Inverse Distance Weighting interpolation technique on a cube

# In this tutorial we will show how to use the Inverse Distance Weighting interpolation technique to deform a cube.
# 
# First of all, we import the required class, the numpy package and we set matplotlib for the notebook.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

from pygem import IDW


# We need to set the deformation parameters: we can set manually, by editing the `IDW` attributes, or we can read them by parsing a file. We remark that it is possible to save the parameters (for example, after set them manually) to a file in order to edit this for the future deformations.

# In[2]:


parameters_file = '../tests/test_datasets/parameters_idw_cube.prm'

idw = IDW()
idw.read_parameters(filename=parameters_file)


# The following is the parameters file for this particular case. The Inverse Distance Weighting section describes the power parameter (see the documentation of the [IDW](http://mathlab.github.io/PyGeM/idw.html) class for more details). As control points we consider the 8 vertices of the cube (the first one is not exactly the vertex), and we move 3 of them. In the Control points section there are all the coordinates of the control points.

# In[3]:


get_ipython().run_line_magic('cat', "'../tests/test_datasets/parameters_idw_cube.prm'")


# Here we create a $10 \times 10 \times 10$ lattice to mimic a cube.

# In[4]:


nx, ny, nz = (10, 10, 10)
mesh = np.zeros((nx * ny * nz, 3))

xv = np.linspace(0, 1, nx)
yv = np.linspace(0, 1, ny)
zv = np.linspace(0, 1, nz)
z, y, x = np.meshgrid(zv, yv, xv)

mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
mesh = mesh.T


# Now we plot the points to see what we are doing.

# In[5]:


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c='blue', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()


# Finally we perform the IDW interpolation using the IDW class.

# In[6]:


new_mesh = idw(mesh)


# We can plot the new points in order to see the deformation. Try different powers to better fit your specific problem.

# In[7]:


fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_mesh[:, 0], new_mesh[:, 1], new_mesh[:, 2], c='red', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

