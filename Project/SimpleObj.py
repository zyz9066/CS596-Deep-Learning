import os
import numpy as np
import pyflann
import binvox_rw
# from models import NearestNeighbor

class SimpleObj(object):
    def __init__(self, bins=200, coef=1.):
        self._bins = bins
        self._coef = coef
        
    def load(self, fileName):
        self._fileName = fileName
        self._vertices = []
        self._faces = []
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
                    self._vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    
                    self._faces.append(face)

            f.close()
        except IOError:
            print(".obj file not found.")
            
        self._vertices = np.array(self._vertices)
            
        self._boundingBox = self._getBoundingBox()    # get the bounding box
        self._center = self._getCenter()
        
    def _getBoundingBox(self):
        """Give a vertices list, get its bounding box.
            Args:
                vertices: vertices list
            Returns:
                bounding box ( xmax, ymax, zmax, xmin, ymin, zmin )
        """
        if len(self._vertices) == 0:
            # vertices list is empty
            return [0,0,0,0,0,0]
    
        xmax, ymax, zmax = np.amax(self._vertices, axis=0)
        xmin, ymin, zmin = np.amin(self._vertices, axis=0)
    
        return np.array([xmax, ymax, zmax, xmin, ymin, zmin])

    def _getCenter(self):
        return np.array([(self._boundingBox[0]+self._boundingBox[3])/2,
                        (self._boundingBox[1]+self._boundingBox[4])/2,
                        (self._boundingBox[2]+self._boundingBox[5])/2])
                        
    def _getVoxels(self):
        x_edge = self._boundingBox[0] - self._boundingBox[3]
        y_edge = self._boundingBox[1] - self._boundingBox[4]
        z_edge = self._boundingBox[2] - self._boundingBox[5]
        edge = max(x_edge, y_edge, z_edge)      # use the max
        voxel_edge = edge/self._bins
    
        x_voxel = int(x_edge/voxel_edge)
        y_voxel = int(y_edge/voxel_edge)
        z_voxel = int(z_edge/voxel_edge)
        self._voxels = np.zeros(shape=(self._bins, self._bins, self._bins)) # Creat a zeros ndarray
        # self._voxel_centers = np.zeros(shape=(x_voxel, y_voxel, z_voxel, 3)) # Creat a zeros ndarray
        # Here, we calculate the start voxel box's center.
        start = self._center - np.array([x_voxel//2*voxel_edge,
                                        y_voxel//2*voxel_edge,
                                        z_voxel//2*voxel_edge])
        # KDtree
        flann = pyflann.FLANN()     # create a FLANN object
        params = flann.build_index(self._vertices, algorithm="kdtree", trees=4)
        
        # nn = NearestNeighbor()     # create a NearestNeighbor object
        # nn.fit(self._vertices)
        
        # calculate the voxel value in iteration
        # if there is a point close to the center, set to 1, otherwise, no changes
        xs, ys, zs = self._voxels.shape
        landmark = self._coef * voxel_edge
        for x in range(x_voxel):
            for y in range(y_voxel):
                for z in range(z_voxel):
                    # for each voxel center
                    voxel_center = np.array([start[0] + x*voxel_edge,
                                              start[1] + y*voxel_edge,
                                              start[2] + z*voxel_edge])
                    result, dists = flann.nn_index(voxel_center, 1, checks=params["checks"])
                    index = result[0]
                    vertex = self._vertices[index,:]  # get nearest neighbor
                    distance = np.sqrt(((vertex - voxel_center) ** 2).sum())
                    # self._voxel_centers[x,y,z] = voxel_center
                    # distance = nn.nn_dist(voxel_center)
                    if distance < landmark:
                        self._voxels[x,y,z] = 1
        
    def saveVoxels(self, outputPath, save_binvox=False):
        """ save voxel
            Save the voxel into file.
        Args:
            outputPath: path to save numpy.
            voxel: numpy.ndarray
        """
        self._getVoxels()
        startPoint = 0
        if self._fileName.rfind("/") != -1:
            startPoint = self._fileName.rfind("/") + 1
    
        fileName = self._fileName[startPoint:self._fileName.rfind('.')]  # cut the format end
        # save npy
        np.save(os.path.join(outputPath, fileName) + ".npy", self._voxels)
        
        if save_binvox:
        # save binvox
            bool_voxel = self._voxels.astype(np.bool)
            binvox = binvox_rw.Voxels(
                data = bool_voxel,
                dims = list(self._voxels.shape),
                translate = [0.0, 0.0, 0.0],
                scale = 1.0,
                axis_order = 'xzy')
            fp = open(os.path.join(outputPath, fileName) + ".binvox", 'wb+')
            fp.truncate()
            binvox.write(fp)
        fp.close()