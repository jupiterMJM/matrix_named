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
    def __init__(self, matrix:np.array, name_matrix:str, axis:list[list[np.array|list, str, str]], denomination_axis:list=None):
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
        for axe in axis:
            if len(axe) != 3:
                raise ValueError("Axis must follow the format [np.array, str, str] for [values, name, unit]")
            self.axis[denomination_axis[i]] = [np.array(axe[0]), *axe[1:]]
        # if len(self.matrix.shape) != 2:
        #     raise ValueError("La matrice doit etre de dimension 2")
        assert len(self.axis.keys()) == len(self.matrix.shape), "Dimensions non cohérentes"	

    def __str__(self):
        message = f"""Matrice : {self.name_matrix}
{"\n".join([f"{key} : {value}" for key, value in self.axis.items()])}
"""
        return message+str(self.matrix)
    
    def apply_transform_on_axis(self, axis:str, fonction:callable, new_name:str=None, new_unit:str=None):
        """
        :param: axis: l'axe surlequel la transformation doit etre appliquée
        :param: fonction: la fonction de transformation
        :note: on ne fait pr l'instant que d'appliquer directement la transformation sur l'axe
        :note: il est necessaire que la transformee soit monotone sur l intervalle considéré, si ce n'est pas le cas, une erreur est levée
        """
        self.axis[axis][0] = np.apply_along_axis(fonction, axis=0, arr=self.axis[axis][0])

    
        if not (np.all(np.diff(self.axis[axis][0]) > 0) or np.all(np.diff(self.axis[axis][0]) < 0)):
            raise ValueError("La transformation n'est pas monotone sur l'intervalle considéré")

        if new_unit is not None:
            self.axis[axis][2] = new_unit
        if new_name is not None:
            self.axis[axis][1] = new_name

    def apply_transform_on_matrix(self, fonction:callable, on_place:bool=True):
        """
        :param: fonction: la fonction de transformation qui doit etre applicable sur une matrice
        """
        if on_place:
            self.matrix = fonction(self.matrix)
        else:
            return fonction(self.matrix)

    def plot(self, title:str=None, ax=None, pcolormesh:bool=True, indicate_max:bool=False, pyqt_plot_item=None, check_for_nan:bool=True):
        """
        :note: on ne fait que de tracer la matrice pour l'instant
        :note: there are 2 ways to plot the matrix : scatter or pcolormesh
            scatter : the matrix is plotted as a scatter plot : each point corresponds to a value of the matrix
                :cons: verry long to plot for big matrix
            pcolormesh : the matrix is plotted as a grid : each cell corresponds to a value of the matrix
                :pros: faster to plot and better for big matrix
            therefore, it is advised to use pcolormesh but we keep scatter in case of
        :param: indicate_max: if True, the maximum value of the matrix will be indicated on the plot
        """
        if pyqt_plot_item is None:
            # produit cartesien des axes
            xv, yv = np.meshgrid(self.axis["abscisse"][0], self.axis["ordonnee"][0])
            # print(self.matrix.reshape(produit(*self.matrix.shape)))
            if ax is None:
                plt.figure()
                if pcolormesh:
                    if check_for_nan:
                        # ignore points where matrix is either nan or inf
                        self.matrix[np.logical_or(np.isinf(xv), np.isinf(yv))] = np.nan
                        xv[np.isinf(xv)] = np.max(xv[np.logical_not(np.isinf(xv))])
                        yv[np.isinf(yv)] = np.max(yv[np.logical_not(np.isinf(yv))])
                        print(xv)
                    plt.pcolormesh(xv, yv, self.matrix, cmap="viridis")
                else:
                    plt.scatter(xv, yv, c=self.matrix, cmap="viridis")
                plt.colorbar(label=self.name_matrix)
                plt.title(title)
                plt.xlabel("{} ({})".format(*self.axis['abscisse'][1:]))
                plt.ylabel("{} ({})".format(*self.axis['ordonnee'][1:]))
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
                ax.set_xlabel("{} ({})".format(*self.axis['abscisse'][1:]))
                ax.set_ylabel("{} ({})".format(*self.axis['ordonnee'][1:]))
                ax.set_title(title)

        else:  # on doit plot avec pyqtgraph
            img_item = pg.ImageItem()
            img_item.setImage(self.matrix.T)        # see notes of setImage in https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html, Transpose because PyQtGraph uses row-major format
            img_item.setColorMap("viridis")
            pyqt_plot_item.setLabel('left', "{} ({})".format(*self.axis['ordonnee'][1:]))
            pyqt_plot_item.setLabel('bottom', "{} ({})".format(*self.axis['abscisse'][1:]))

            x = self.axis['abscisse'][0].flatten()
            y = self.axis['ordonnee'][0].flatten()
            X, Y = np.meshgrid(x, y, indexing="ij")  # Create grid
            nan_mask = np.isnan(X) | np.isnan(Y) #| np.isnan(self.matrix) | np.isinf(X) | np.isinf(Y) | np.isinf(self.matrix)
            print(nan_mask.shape, self.matrix.shape)
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

            x_axis.setTicks([[(i, str(i)) for i in range(len(self.axis['abscisse'][0]))]])
            y_axis.setTicks([[(i, str(i)) for i in range(len(self.axis['ordonnee'][0]))]])
            pyqt_plot_item.addItem(img_item)
            # pyqt_plot_item.autoRange()
            # pyqt_plot_item.getViewBox().autoRange()


    def plot_3d(self, title:str=None):
        """
        :note: on ne fait que de tracer la matrice pour l'instant
        """
        # produit cartesien des axes
        xv, yv = np.meshgrid(self.axis["abscisse"][0], self.axis["ordonnee"][0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xv, yv, self.matrix)
        ax.set_xlabel("{} ({})".format(*self.axis['abscisse'][1:]))
        ax.set_ylabel("{} ({})".format(*self.axis['ordonnee'][1:]))
        ax.set_zlabel(self.name_matrix)
        plt.title(title)
        plt.legend() 

    
    def crop_matrix(self, axis:str, bornes:tuple, on_place:bool=True):
        """
        :param: axis: l'axe sur lequel on doit croper
        :param: bornes: les bornes de l'axe
        """
        # on recupere les indices des bornes pour l'axe en question
        indices = np.logical_and(self.axis[axis][0] >= bornes[0], self.axis[axis][0] <= bornes[1])
        
        # on recupere la nouvelle matrice UNIQUEMENT sur l'axe en question
        
        if axis == "abscisse":
            new_matrix = self.matrix[:, indices.flatten()]
        else:
            new_matrix = self.matrix[indices.flatten(), :]
        # on recupere le nouvel axe
        new_axis = [self.axis[axis][0][indices], *self.axis[axis][1:]]
        new_axis[0] = new_axis[0].reshape(1, -1)
        if on_place:
            # on met a jour la matrice
            self.matrix = new_matrix
            # on met a jour l'axe
            self.axis[axis] = new_axis
        else:
            # on cree la nouvelle matrice
            if axis == "abscisse":
                new_matrix = MatrixeNamed(new_matrix, self.name_matrix, abscisse=new_axis, ordonnee=self.axis["ordonnee"])
            else:
                new_matrix = MatrixeNamed(new_matrix, self.name_matrix, abscisse=self.axis["abscisse"], ordonnee=new_axis)
            return new_matrix
        
    def sum_along_axis(self, axis:str):
        """
        :param: axis: l'axe sur lequel on doit sommer (cad l'indice que l'on fait varier)
        """
        if axis == "abscisse":
            somme = np.sum(self.matrix, axis=1)
            axe_retour = "ordonnee"
        else:
            somme = np.sum(self.matrix, axis=0)
            axe_retour = "abscisse"
        return somme, self.axis[axe_retour][0]
    
    def iloc(self, axis:str, value:int):
        """
        :param: axis: l'axe sur lequel on doit chercher
        :param: value: la valeur à chercher
        :note: on ne fait que de l'interpolation linéaire pour l'instant
        :note: inspirer de la fonction pandas.DataFrame.iloc
        """
        pass # TODO

    def loc(self, axis:str, value:float, take_the_closest:bool=True):
        """
        :param: axis: l'axe sur lequel on doit chercher
        :param: value: la valeur à chercher
        :note: on ne fait que de l'interpolation linéaire pour l'instant
        :note: inspirer de la fonction pandas.DataFrame.loc
        :note: si axis=="abscisse", on renvoie la colonne correspondante
        """
        # on recupere l'indice le plus proche
        if take_the_closest:
            index = np.argmin(np.abs(self.axis[axis][0] - value))
        else:
            index = np.argwhere(self.axis[axis]==value)
            print("Attention aux erreurs!!!")
        
        if axis == "abscisse":
            # on recupere la colonne correspondante
            return self.matrix[:, index]
        elif axis == "ordonnee":
            # on recupere la ligne correspondante
            return self.matrix[index, :]
        else:
            raise ValueError("L'axe doit etre soit 'abscisse' soit 'ordonnee'")
    


def interpolate_matrix(matrix:MatrixeNamed, axis:str):
    """
    :param: matrix: la matrice à interpoler
    :param: axis: l'axe sur lequel on doit interpoler
    :note: on ne fait que de l'interpolation linéaire pour l'instant
    :note: l'interpolation sera faite en prenant le plus petit pas d'axe
    :warning: not tested yet (bcs not needed)
    """
    # on recupere le plus petit pas d'axe
    step = np.min(np.abs(np.diff(matrix.axis[axis][0])))
    # on recupere les bornes de l'axe
    bornes = [matrix.axis[axis][0][0], matrix.axis[axis][0][-1]]
    print(step)
    # on cree le nouvel axe
    new_axis = np.arange(bornes[0], bornes[1], step)
    # on cree la nouvelle matrice
    new_matrix = np.zeros((len(new_axis), matrix.matrix.shape[1]))
    # on interpole
    for i in range(matrix.matrix.shape[1]):
        new_matrix[:, i] = np.interp(new_axis, matrix.axis[axis][0], matrix.matrix[:, i])
    # on cree la nouvelle matrice
    new_matrix = MatrixeNamed(new_matrix, matrix.name_matrix, abscisse=[new_axis, *matrix.axis[axis][1:]], ordonnee=matrix.axis["ordonnee"])
    return new_matrix




if __name__ == "__main__" :
    test = MatrixeNamed(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), "matrice test", abscisse=[[1, 2, 3], "Duree", "s"], ordonnee=[[4, 5, 6] , "Distance", "m"])
    test.crop_matrix("abscisse", (1.5, 2.5))
    print(test.matrix)