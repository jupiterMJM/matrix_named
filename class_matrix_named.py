"""
on cree ici une classe correspondant à une matrice ayant des axes.
l'objectif premier est de pouvoir effectuer des transformees facilement sur ces matrices
:rq: ce fichier est extrait du projet numéro 1
:rq: cette version de classe matrix_named est plus avancée que la version de base
"""


# importation des modules
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg


class MatrixeNamed:
    def __init__(self, matrix:np.array, name_matrix:str, axis:list[list[list, str, str]], denomination_axis:list=None):
        """
        :param: matrix: np.array, the values of the numpy matrix
        :param: name_matrix: str, the name of the matrix (what it represents)
        :param: axis: list[list[np.array, str, str]], the axes of the matrix, BUT it is ordered : for a 2D matrix, the first axis is the abscisse and the second is the ordinate
        :param: denomination_axis: list[str], the denomination of the axis, if not specified, the default is axis_1, axis_2, ...
                WHAT IS IT : there is two way to talk about an axis : the "physical" way aka what it means in reality and the "mathematical" way aka abscissa, ordinate, ...
                the name of the axis is the "physical" way, and the denomination is the "mathematical" way1
        :note: shape must be coherent (otherwise an error is raised)
        :note: ORIENTATION : the orientation must be logical, i.e. we imagine that the matrix is an image so the axes are the axes of the image from left to right and from bottom to top
        :note: denomination of the axis : unless specified the name of the axis are as follow : axis_1, axis_2, ...
        """
        self.name_matrix = name_matrix
        self.matrix = matrix

        if denomination_axis is None:
            denomination_axis = [f"axis_{i}" for i in range(len(axis))]
        assert len(denomination_axis) == len(axis), "Denomination of axis must be coherent with the number of axis"

        self.axis = dict()
        for i, axe in enumerate(axis):
            if len(axe) != 3:
                raise ValueError("Axis must follow the format [np.array, str, str] for [values, name, unit]")
            self.axis[denomination_axis[i]] = [np.array(axe[0]), *axe[1:]]
        # if len(self.matrix.shape) != 2:
        #     raise ValueError("La matrice doit etre de dimension 2")
        assert len(self.axis.keys()) == len(self.matrix.shape), "Dimensions non cohérentes"	

    def __str__(self):
        """
        just to print some info about the matrix
        """
        message = f"""Matrice : {self.name_matrix}
{"\n".join([f"{key} : {value}" for key, value in self.axis.items()])}
"""
        return message+str(self.matrix)
    
    def apply_transform_on_axis(self, axis:str, fonction:callable, new_name:str=None, new_unit:str=None, force_apply:bool=False):
        """
        :param: axis : the axis on which the transformation must be applied
        :param: fonction : the transformation function
        :param: new_name : the new name of the axis
        :param: new_unit : the new unit of the axis
        :param: force_apply : if True, the transformation is applied even if it is not monotone (not recommended, because it may crashes afterward)
        :note: we only apply the transformation directly on the axis
        :note: it is necessary that the transformation be monotone on the considered interval, if not an error is raised
        :note: can be used as a conversion function on the axis
        """
        self.axis[axis][0] = np.apply_along_axis(fonction, axis=0, arr=self.axis[axis][0])

    
        if not force_apply and not (np.all(np.diff(self.axis[axis][0]) > 0) or np.all(np.diff(self.axis[axis][0]) < 0)):
            raise ValueError("The tranformation is not monotone on the considered interval")

        if new_unit is not None:
            self.axis[axis][2] = new_unit
        if new_name is not None:
            self.axis[axis][1] = new_name

    def apply_transform_on_matrix(self, fonction:callable, on_place:bool=True):
        """
        :param: fonction : the transformation function that will be applied to the matrix 
        :note: for now, the function must be appliable on a random matrix numpy
        """
        if on_place:
            self.matrix = fonction(self.matrix)
        else:
            return fonction(self.matrix)

    def plot(self, title:str=None, axis_to_be_plot:list[str]=None, ax=None, pcolormesh:bool=True, indicate_max:bool=False, pyqt_plot_item=None, check_for_nan:bool=True):
        """
        :param: title: the title of the plot
        :param: axis_to_be_plot: the axis to be plotted, if None, the first two axes are plotted
        :param: ax: the axis on which the plot will be done (if you want to insert the plot in a subplot)
        :param: pcolormesh: if True, the matrix will be plotted as a grid, if False, the matrix will be plotted as a scatter plot, not recommended for big matrix
        :param: indicate_max: if True, the maximum value of the matrix will be indicated on the plot
        :param: pyqt_plot_item: if not None, the plot will be done with pyqtgraph, the plot will be done on the pyqt_plot_item
        :param: check_for_nan: if True, the points where the matrix is nan or inf will be ignored
        :note: for now, we only plot the matrix
        :note: there are 2 ways to plot the matrix : scatter or pcolormesh
            scatter : the matrix is plotted as a scatter plot : each point corresponds to a value of the matrix
                :cons: verry long to plot for big matrix
            pcolormesh : the matrix is plotted as a grid : each cell corresponds to a value of the matrix
                :pros: faster to plot and better for big matrix
            therefore, it is advised to use pcolormesh but we keep scatter in case of
        :param: indicate_max: if True, the maximum value of the matrix will be indicated on the plot
        """
        if len(self.matrix.shape) == 1:
            # we plot directly
            plt.figure()
            plt.plot(self.axis[self.axis.keys[0]][0], self.matrix)
            plt.xlabel("{} ({})".format(*self.axis[self.axis.keys[0]][1:]))
            plt.ylabel(self.name_matrix)
            plt.title(title)
            return

        if axis_to_be_plot is None:
            axis1, axis2 = list(self.axis.keys())[:2]
        else:
            axis1, axis2 = axis_to_be_plot

        if pyqt_plot_item is None:      # we plot with matplotlib
            # produit cartesien des axes
            xv, yv = np.meshgrid(self.axis[axis1][0], self.axis[axis2][0])
            if ax is None:
                plt.figure()
                if pcolormesh:
                    if check_for_nan:   # TODO looks like there may be an error here
                        # ignore points where matrix is either nan or inf
                        self.matrix[np.logical_or(np.isinf(xv), np.isinf(yv))] = np.nan
                        xv[np.isinf(xv)] = np.max(xv[np.logical_not(np.isinf(xv))])
                        yv[np.isinf(yv)] = np.max(yv[np.logical_not(np.isinf(yv))])
                        # print(xv)
                    plt.pcolormesh(xv, yv, self.matrix, cmap="viridis")
                else:
                    plt.scatter(xv, yv, c=self.matrix, cmap="viridis")
                plt.colorbar(label=self.name_matrix)
                plt.title(title)
                plt.xlabel("{} ({})".format(*self.axis[axis1][1:]))
                plt.ylabel("{} ({})".format(*self.axis[axis2][1:]))
                if indicate_max:
                    plt.scatter(xv.flatten()[np.argmax(self.matrix)], yv.flatten()[np.argmax(self.matrix)], color="red", marker="x")
            else:
                if pcolormesh:
                    self.matrix[np.logical_or(np.isinf(xv), np.isinf(yv))] = np.nan
                    xv[np.isinf(xv)] = np.max(xv[np.logical_not(np.isinf(xv))])
                    yv[np.isinf(yv)] = np.max(yv[np.logical_not(np.isinf(yv))])
                    ax.pcolormesh(xv, yv, self.matrix, cmap="viridis")
                else:
                    ax.scatter(xv, yv, c=self.matrix, cmap="viridis")
                ax.set_xlabel("{} ({})".format(*self.axis[axis1][1:]))
                ax.set_ylabel("{} ({})".format(*self.axis[axis2][1:]))
                ax.set_title(title)

        else:  # on doit plot avec pyqtgraph
            img_item = pg.ImageItem()
            img_item.setImage(self.matrix.T)        # see notes of setImage in https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html, Transpose because PyQtGraph uses row-major format
            img_item.setColorMap("viridis")
            pyqt_plot_item.setLabel('left', "{} ({})".format(*self.axis[axis2][1:]))
            pyqt_plot_item.setLabel('bottom', "{} ({})".format(*self.axis[axis1][1:]))

            x = self.axis[axis1][0].flatten()
            y = self.axis[axis2][0].flatten()
            X, Y = np.meshgrid(x, y, indexing="ij")  # Create grid
            nan_mask = np.isnan(X) | np.isnan(Y) #| np.isnan(self.matrix) | np.isinf(X) | np.isinf(Y) | np.isinf(self.matrix)
            # print(nan_mask.shape, self.matrix.shape)
            X[X==np.nan] = 0
            Y[X==np.nan] = 0
            self.matrix[nan_mask.T] = np.nan
            dx = np.diff(x)  # X bin widths
            dy = np.diff(y)  # Y bin widths
            scale_x = dx.mean()  # Approximate uniform width
            scale_y = dy.mean()  # Approximate uniform height
            img_item.setTransform(pg.QtGui.QTransform(scale_x, 0, 0, scale_y, x[0], y[0]))

            # Get axis objects
            x_axis = pyqt_plot_item.getAxis('bottom')
            y_axis = pyqt_plot_item.getAxis('left')

            x_axis.setTicks([[(i, str(i)) for i in range(len(self.axis[axis1][0]))]])
            y_axis.setTicks([[(i, str(i)) for i in range(len(self.axis[axis2][0]))]])
            pyqt_plot_item.addItem(img_item)
            # pyqt_plot_item.autoRange()
            # pyqt_plot_item.getViewBox().autoRange()


    def plot_3d(self, title:str=None, axis_to_be_plot:list[str]=None):
        """
        :param: title: the title of the plot
        :param: axis_to_be_plot: the axis to be plotted, if None, the first two axes are plotted
        :note: if 3 axis are given, the order is the following : x, y, z and the values of the matrix are in the colormap
            but if 2 are given, the order is the following : x, y and the values of the matrix are in the z axis
            by default we consider that we want only the first two axes
        :note: for now, we only plot the matrix
        :note: this function allows to plot the matrix in 3D (instead of a colormap)
        """
        # TODO better deal with the axis and multi dimensionnal data
        # produit cartesien des axes
        if axis_to_be_plot is None:
            axis1, axis2 = list(self.axis.keys())[:2]

        if len(axis_to_be_plot) == 2:
            axis1, axis2 = axis_to_be_plot
            xv, yv = np.meshgrid(self.axis[axis1][0], self.axis[axis2][0])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xv, yv, self.matrix)
            ax.set_xlabel("{} ({})".format(*self.axis['abscisse'][1:]))
            ax.set_ylabel("{} ({})".format(*self.axis['ordonnee'][1:]))
            ax.set_zlabel(self.name_matrix)
            plt.title(title)
            plt.legend() 
        elif len(axis_to_be_plot) == 3:
            axis1, axis2, axis3 = axis_to_be_plot
            xv, yv, zv = np.meshgrid(self.axis[axis1][0], self.axis[axis2][0], self.axis[axis3][0])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xv, yv, zv, c=self.matrix)
            ax.set_xlabel("{} ({})".format(*self.axis[axis1][1:]))
            ax.set_ylabel("{} ({})".format(*self.axis[axis2][1:]))
            ax.set_zlabel("{} ({})".format(*self.axis[axis3][1:]))
            plt.title(title)
            plt.legend()
        else:
            raise ValueError("The number of axis to be plot3D must be 2 or 3")

    
    def crop_matrix(self, axis:str, bornes:tuple, on_place:bool=True):
        """
        :param: axis: the axis on which we must crop the matrix
        :param: bornes: the bounds of the axis
        :param: on_place: if True, the matrix is cropped in place, if False, a new matrix is returned
        :note: the matrix is cropped on the axis in question
        """
        # TODO deal with multidmensionnal data
        assert type(axis) == str and axis in self.axis.keys(), "The axis must be a string and must be in the axis of the matrix"
        assert len(bornes) == 2, "The bounds must be a tuple of 2 elements"
        bornes = np.sort(bornes)

        # on recupere les indices des bornes pour l'axe en question
        indices = np.logical_and(self.axis[axis][0] >= bornes[0], self.axis[axis][0] <= bornes[1])
        
        # on recupere la nouvelle matrice UNIQUEMENT sur l'axe en question
        num_axis = list(self.axis.keys()).index(axis)
        if num_axis == len(self.matrix.shape)-1:
            commande = f"self.matrix[{':, '*max(num_axis-2, 0)}indices.flatten(), :]"
        elif num_axis == len(self.matrix.shape)-2:
            commande = f"self.matrix[{':, '*max(num_axis-1, 0)} :, indices.flatten()]"
        else:
            commande = f"self.matrix[{':, '*num_axis}indices.flatten()]"
        new_matrix = eval(commande)     # maybe not the more beautiful but idk how to do it better
        # if axis == "abscisse":
        #     new_matrix = self.matrix[:, indices.flatten()]
        # else:
        #     new_matrix = self.matrix[indices.flatten(), :]
        # on recupere le nouvel axe
        new_axis = [self.axis[axis][0][indices], *self.axis[axis][1:]]
        new_axis[0] = new_axis[0].reshape(1, -1)
        if on_place:
            # on met a jour la matrice
            self.matrix = new_matrix
            # on met a jour l'axe
            self.axis[axis] = new_axis
        else:
            # let s create the new matrix
            all_axis = list(self.axis.values())
            if num_axis == len(self.axis.keys())-1:
                axis_construction = all_axis[:num_axis] + [new_axis]
            else:
                axis_construction = all_axis[:num_axis] + [new_axis] + all_axis[num_axis+1:]
            new_matrix = MatrixeNamed(new_matrix, self.name_matrix, axis_construction, denomination_axis=list(self.axis.keys()))
            # if axis == "abscisse":
            #     new_matrix = MatrixeNamed(new_matrix, self.name_matrix, abscisse=new_axis, ordonnee=self.axis["ordonnee"])
            # else:
            #     new_matrix = MatrixeNamed(new_matrix, self.name_matrix, abscisse=self.axis["abscisse"], ordonnee=new_axis)
            return new_matrix
        
    def sum_along_axis(self, axis:str) -> tuple[np.array, np.array]:
        """
        :param: axis: the axis along which we have to sum
        :note: we sum the matrix along the axis in question
        :return: the sum and the axis along which the sum was made
        """
        if len(self.matrix.shape) != 2:
            raise ValueError("Not implemented yet")
            # TODO deal with multidmensionnal data
        if axis == "abscisse":
            somme = np.sum(self.matrix, axis=1)
            axe_retour = "ordonnee"
        else:
            somme = np.sum(self.matrix, axis=0)
            axe_retour = "abscisse"
        return somme, self.axis[axe_retour][0]
    
    def iloc(self, axis:str, value:int):
        raise Exception("Not implemented yet") # TODO

    def loc(self, axis:str, value:float, take_the_closest:bool=True):
        """
        :param: axis: the axis on which we must search
        :param: value: the value to search
        :param: take_the_closest: if True, we take the closest value, if False, we take the exact value
        :return: the line or the column corresponding to the value
        :note: inspired by pandas
        """
        # TODO deal with multidmensionnal data
        # on recupere l'indice le plus proche
        if take_the_closest:
            index = np.argmin(np.abs(self.axis[axis][0] - value))
        else:
            index = np.argwhere(self.axis[axis]==value) # index is used in the eval function
            print("Attention aux erreurs!!!")

        # on recupere la nouvelle matrice UNIQUEMENT sur l'axe en question
        num_axis = list(self.axis.keys()).index(axis)
        if num_axis == len(self.matrix.shape)-1:
            commande = f"self.matrix[{':, '*max(num_axis-2, 0)}index, :]"
        elif num_axis == len(self.matrix.shape)-2:
            commande = f"self.matrix[{':, '*max(num_axis-1, 0)} :, index]"
        else:
            commande = f"self.matrix[{':, '*num_axis}index]"
        
        new_matrix = eval(commande)     # maybe not the more beautiful but idk how to do it better

        return new_matrix
    




if __name__ == "__main__" :
    test = MatrixeNamed(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), "matrice test", [[[1, 2, 3], "Duree", "s"], [[4, 5, 6] , "Distance", "m"]])
    test.plot()
    plt.show()