import pyvista as pv
import numpy as np
from LoopStructural.datatypes import VectorPoints, ValuePoints
from LoopStructural.modelling.features import BaseFeature

# from LoopStructural.modelling.features.fault import FaultSegment
from LoopStructural import GeologicalModel
from LoopStructural.utils import getLogger
from typing import Union, Optional, List

logger = getLogger(__name__)


class Loop3DView(pv.Plotter):
    def __init__(self, model=None, background='white', *args, **kwargs):
        """Loop3DView is a subclass of pyvista. Plotter that is designed to
        interface with the LoopStructural geological modelling package.

        Parameters
        ----------
        model : GeologicalModel, optional
            A loopstructural model used as reference for some methods, by default None
        background : str, optional
            colour for the background, by default 'white'
        """
        super().__init__(*args, **kwargs)
        self.set_background(background)
        self.model = model

    def _check_model(self, model: GeologicalModel) -> GeologicalModel:
        """helper method to assign a geological model"""
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("No model provided")
        return model

    def plot_surface(
        self,
        geological_feature: BaseFeature,
        value: Optional[Union[float, int]] = None,
        paint_with: Optional[BaseFeature] = None,
        colour: Optional[str] = "red",
        cmap: Optional[str] = None,
        opacity: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        pyvista_kwargs: dict = {},
        scalar_bar: bool = False,
        slicer: bool = False,
    ):
        """Add an isosurface of a geological feature to the model

        Parameters
        ----------
        geological_feature : BaseFeature
            The geological feature to plot
        value : Optional[Union[float, int, List[float]]], optional
            isosurface value, or list of values, by default average value of feature
        paint_with : Optional[BaseFeature], optional
            Paint the surface with the value of another geological feature, by default None
        colour : Optional[str], optional
            colour of the surface, by default "red"
        cmap : Optional[str], optional
            matplotlib colourmap, by default None
        opacity : Optional[float], optional
            opacity of the surface, by default None
        vmin : Optional[float], optional
            minimum value of the colourmap, by default None
        vmax : Optional[float], optional
            maximum value of the colourmap, by default None
        pyvista_kwargs : dict, optional
            other parameters passed to Plotter.add_mesh, by default {}
        """

        surfaces = geological_feature.surfaces(value)
        meshes = []
        for surface in surfaces:
            s = surface.vtk
            if paint_with is not None:
                clim = [paint_with.min(), paint_with.max()]
                if vmin is not None:
                    clim[0] = vmin
                if vmax is not None:
                    clim[1] = vmax
                pyvista_kwargs["clim"] = clim
                pts = np.copy(surface.vertices)
                if self.model is not None:
                    pts = self.model.scale(pts)
                scalars = paint_with(pts)
                s["values"] = scalars
                s.set_active_scalars("values")
                colour = None
            meshes.append(s)
        if slicer:
            self.add_mesh_clip_plane(
                pv.MultiBlock(meshes).combine(),
                color=colour,
                cmap=cmap,
                opacity=opacity,
                **pyvista_kwargs,
            )
        else:
            self.add_mesh(
                pv.MultiBlock(meshes).combine(),
                color=colour,
                cmap=cmap,
                opacity=opacity,
                **pyvista_kwargs,
            )
        if paint_with is not None and not scalar_bar:
            self.remove_scalar_bar('values')

    def plot_scalar_field(
        self,
        geological_feature,
        cmap="viridis",
        vmin=None,
        vmax=None,
        opacity=None,
        pyvista_kwargs={},
        scalar_bar: bool = False,
        slicer=False,
    ):
        volume = geological_feature.scalar_field()
        if vmin is not None:
            pyvista_kwargs["clim"][0] = vmin
        if vmax is not None:
            pyvista_kwargs["clim"][1] = vmax
        if slicer:
            self.add_mesh_clip_plane(volume, cmap=cmap, opacity=opacity, **pyvista_kwargs)
        else:
            self.add_mesh(volume, cmap=cmap, opacity=opacity, **pyvista_kwargs)
        if not scalar_bar:
            self.remove_scalar_bar(geological_feature.name)

    def plot_block_model(
        self,
        cmap=None,
        model=None,
        pyvista_kwargs={},
        scalar_bar: bool = False,
        slicer: bool = False,
        threshold: Optional[Union[float, List[float]]] = None,
    ):
        model = self._check_model(model)

        block, codes = model.get_block_model()
        cmap = self._build_stratigraphic_cmap(model)
        if "clim" not in pyvista_kwargs:
            pyvista_kwargs["clim"] = (np.min(block['stratigraphy']), np.max(block['stratigraphy']))
        if threshold is not None:
            if isinstance(threshold, float):
                block = block.threshold(threshold)
            elif isinstance(threshold, (list, tuple, np.ndarray)) and len(threshold) == 2:
                block = block.threshold((threshold[0], threshold[1]))
        if slicer:
            self.add_mesh_clip_plane(block, cmap=cmap, **pyvista_kwargs)
        else:
            self.add_mesh(block, cmap=cmap, **pyvista_kwargs)
        if not scalar_bar:
            self.remove_scalar_bar('stratigraphy')

    def plot_fault_displacements(
        self,
        fault_list=None,
        bounding_box=None,
        model=None,
        cmap="rainbow",
        pyvista_kwargs={},
        scalar_bar: bool = False,
    ):
        if fault_list is None:
            model = self._check_model(model)
            fault_list = model.faults
        if bounding_box is None:
            model = self._check_model(model)
            bounding_box = model.bounding_box
        pts = bounding_box.regular_grid()
        displacement_value = np.zeros(pts.shape[0])
        for f in fault_list:
            disp = f.displacementfeature.evaluate_value(bounding_box.vtk.points)
            displacement_value[~np.isnan(disp)] += disp[~np.isnan(disp)]
        volume = bounding_box.vtk
        volume['displacement'] = displacement_value
        self.add_mesh(volume, cmap=cmap, **pyvista_kwargs)
        if not scalar_bar:
            self.remove_scalar_bar('displacement')

    def _build_stratigraphic_cmap(self, model):
        try:
            import matplotlib.colors as colors

            colours = []
            boundaries = []
            data = []
            for g in model.stratigraphic_column.keys():
                if g == "faults":
                    continue
                for v in model.stratigraphic_column[g].values():
                    data.append((v["id"], v["colour"]))
                    colours.append(v["colour"])
                    boundaries.append(v["id"])  # print(u,v)
            cmap = colors.ListedColormap(colours).colors
        except ImportError:
            logger.warning("Cannot use predefined colours as I can't import matplotlib")
            cmap = "tab20"
        return cmap

    def plot_model_surfaces(
        self,
        strati=True,
        faults=True,
        cmap=None,
        model=None,
        fault_colour="black",
        paint_with=None,
        displacement_cmap=None,
        pyvista_kwargs={},
        scalar_bar: bool = False,
    ):
        model = self._check_model(model)

        if strati:
            strati_surfaces = []
            surfaces = model.get_stratigraphic_surfaces()
            if cmap is None:
                cmap = self._build_stratigraphic_cmap(model)
                print(cmap)
            for s in surfaces:
                strati_surfaces.append(s.vtk)
            self.add_mesh(pv.MultiBlock(strati_surfaces), cmap=cmap, **pyvista_kwargs)
            if not scalar_bar:
                self.remove_scalar_bar()
        if faults:
            faults = model.get_fault_surfaces()
            for f in faults:
                self.add_mesh(f.vtk, color=fault_colour, **pyvista_kwargs)

    def plot_vector_field(self, geological_feature, scale=1.0, pyvista_kwargs={}):
        vectorfield = geological_feature.vector_field()
        self.add_mesh(vectorfield.vtk(scale=scale), **pyvista_kwargs)
        pass

    def plot_data(
        self,
        feature,
        value=True,
        vector=True,
        scale=10,
        geom="arrow",
        pyvista_kwargs={},
    ):
        for d in feature.get_data():
            if isinstance(d, ValuePoints):
                if value:
                    self.add_mesh(d.vtk(), **pyvista_kwargs)
            if isinstance(d, VectorPoints):
                if vector:
                    self.add_mesh(d.vtk(geom=geom, scale=scale), **pyvista_kwargs)

    def plot_fold(self, fold, pyvista_kwargs={}):

        pass

    def plot_fault(
        self,
        fault,
        surface=True,
        slip_vector=True,
        displacement_scale_vector=True,
        fault_volume=True,
        vector_scale=200,
        pyvista_kwargs={},
    ):
        if surface:

            surface = fault.surfaces([0])[0]
            self.add_mesh(surface.vtk, **pyvista_kwargs)
        if slip_vector:

            vectorfield = fault[1].vector_field()
            self.add_mesh(
                vectorfield.vtk(
                    scale=vector_scale,
                    scale_function=(
                        fault.displacementfeature.evaluate_value
                        if displacement_scale_vector
                        else None
                    ),
                ),
                **pyvista_kwargs,
            )
        if fault_volume:
            volume = fault.displacementfeature.scalar_field()
            volume.threshold(0.0)
            self.add_mesh(volume, **pyvista_kwargs)

    def rotate(self, angles: np.ndarray):
        """Rotate the camera by the given angles
        order is roll, azimuth, elevation as defined by
        pyvista

        Parameters
        ----------
        angles : np.ndarray
            roll, azimuth, elevation
        """
        self.camera.roll += angles[0]
        self.camera.azimuth += angles[1]
        self.camera.elevation += angles[2]

    def display(self):
        self.show(interactive=False)
