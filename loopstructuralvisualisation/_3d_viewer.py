import pyvista as pv
import numpy as np
from LoopStructural.datatypes import VectorPoints, ValuePoints
from LoopStructural.modelling.features import BaseFeature

from LoopStructural.modelling.features.fault import FaultSegment
from LoopStructural.datatypes import BoundingBox
from LoopStructural import GeologicalModel
from LoopStructural.utils import getLogger
from typing import Union, Optional, List, Callable
from ._colours import random_colour

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
        self.objects = {}

    def add_mesh(self, *args, **kwargs):
        if 'name' not in kwargs:
            name = 'unnamed_object'
            name = self.increment_name(name)
            kwargs['name'] = name
            logger.warning(
                f'No name provided, using {name}. Pass name argument to add_mesh to remove this error'
            )
        return super().add_mesh(*args, **kwargs)

    def increment_name(self, name):
        parts = name.split('_')
        if len(parts) == 1:
            name = name + '_1'
        while name in self.actors:
            parts = name.split('_')
            try:
                parts[-1] = str(int(parts[-1]) + 1)
            except ValueError:
                parts.append('1')
            name = '_'.join(parts)
        return name

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
        name: Optional[str] = None,
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
        name : Optional[str], optional
            name of the object, by default None
        slicer : bool, optional
            If an interactive plane slicing tool should be added, by default False
        scalar_bar : bool, optional
            Whether to show the scalar bar, by default False
        """

        if name is None:
            name = geological_feature.name + '_surfaces'
        name = self.increment_name(name)  # , 'surface')

        surfaces = geological_feature.surfaces(value)
        meshes = []
        for surface in surfaces:
            s = surface.vtk()
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
        mesh = pv.MultiBlock(meshes).combine()
        actor = None
        try:

            if slicer:
                actor = self.add_mesh_clip_plane(
                    mesh,
                    color=colour,
                    cmap=cmap,
                    opacity=opacity,
                    name=name,
                    **pyvista_kwargs,
                )
            else:
                actor = self.add_mesh(
                    mesh,
                    color=colour,
                    cmap=cmap,
                    opacity=opacity,
                    name=name,
                    **pyvista_kwargs,
                )

        except ValueError:
            logger.warning("No surfaces to plot")
        if paint_with is not None and not scalar_bar:
            self.remove_scalar_bar('values')
        return actor

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
        name=None,
    ):
        if name is None:
            name = geological_feature.name + '_scalar_field'
        name = self.increment_name(name)  # , 'scalar_field')

        volume = geological_feature.scalar_field().vtk()
        if vmin is not None:
            pyvista_kwargs["clim"][0] = vmin
        if vmax is not None:
            pyvista_kwargs["clim"][1] = vmax
        if slicer:
            actor = self.add_mesh_clip_plane(
                volume, cmap=cmap, opacity=opacity, name=name, **pyvista_kwargs
            )
        else:
            actor = self.add_mesh(volume, cmap=cmap, opacity=opacity, name=name, **pyvista_kwargs)
        if not scalar_bar:
            self.remove_scalar_bar(geological_feature.name)
        return actor

    def plot_block_model(
        self,
        cmap=None,
        model=None,
        pyvista_kwargs={},
        scalar_bar: bool = False,
        slicer: bool = False,
        threshold: Optional[Union[float, List[float]]] = None,
        name: Optional[str] = None,
    ):
        """Plot a voxel model where the stratigraphic id is the active scalar.
        It will use the colours defined in the stratigraphic column of the model
        unless a cmap is provided.
        Min/max range of cmap are defined by the min/max values of the stratigraphic ids or if
        clim is provided in pyvista_kwargs

        Parameters
        ----------
        cmap : str, optional
            matplotlib cmap string, by default None
        model : GeologicalModel, optional
            the model to pass if it is not the active geologicalmodel, by default None
        pyvista_kwargs : dict, optional
            additional arguments to be passed to pyvista add_mesh, by default {}
        scalar_bar : bool, optional
            whether show/hide the scalar bar, by default False
        slicer : bool, optional
            If an interactive plane slicing tool should be added, by default False
        threshold : Optional[Union[float, List[float]]], optional
            Whether to threshold values of the stratigraphy. Uses same syntax as pyvista threshold., by default None
        """
        model = self._check_model(model)
        if name is None:
            name = 'block_model'
        name = self.increment_name(name)  # , 'block_model')
        block, codes = model.get_block_model()
        block = block.vtk()
        block.set_active_scalars('stratigraphy')
        actor = None
        if cmap is None:
            cmap = self._build_stratigraphic_cmap(model)
        if "clim" not in pyvista_kwargs:
            pyvista_kwargs["clim"] = (np.min(block['stratigraphy']), np.max(block['stratigraphy']))
        if threshold is not None:
            if isinstance(threshold, float):
                block = block.threshold(threshold)
            elif isinstance(threshold, (list, tuple, np.ndarray)) and len(threshold) == 2:
                block = block.threshold((threshold[0], threshold[1]))
        if slicer:
            actor = self.add_mesh_clip_plane(block, cmap=cmap, name=name, **pyvista_kwargs)
        else:
            actor = self.add_mesh(block, cmap=cmap, name=name, **pyvista_kwargs)

        if not scalar_bar:
            self.remove_scalar_bar('stratigraphy')
        return actor

    def plot_fault_displacements(
        self,
        fault_list: Optional[List[FaultSegment]] = None,
        bounding_box: Optional[BoundingBox] = None,
        model=None,
        cmap="rainbow",
        pyvista_kwargs={},
        scalar_bar: bool = False,
        name: Optional[str] = None,
    ):
        """Plot the dispalcement magnitude for faults in the model
        on a voxel block

        Parameters
        ----------
        fault_list : _type_, optional
            list of faults to plot the model, by default None
        bounding_box : _type_, optional
            _description_, by default None
        model : _type_, optional
            _description_, by default None
        cmap : str, optional
            _description_, by default "rainbow"
        pyvista_kwargs : dict, optional
            _description_, by default {}
        scalar_bar : bool, optional
            _description_, by default False
        """
        if name is None:
            name = 'fault_displacement'
        name = self.increment_name(name)  # , 'fault_displacement_map')
        if fault_list is None:
            model = self._check_model(model)
            fault_list = model.faults
        if bounding_box is None:
            model = self._check_model(model)
            bounding_box = model.bounding_box
        pts = bounding_box.regular_grid()
        displacement_value = np.zeros(pts.shape[0])
        for f in fault_list:
            disp = f.displacementfeature.evaluate_value(bounding_box.vtk().points)
            displacement_value[~np.isnan(disp)] += disp[~np.isnan(disp)]
        volume = bounding_box.vtk()
        volume['displacement'] = displacement_value
        actor = self.add_mesh(volume, cmap=cmap, **pyvista_kwargs)
        if not scalar_bar:
            self.remove_scalar_bar('displacement')
        return actor

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
                    if not isinstance(v['colour'], str):
                        try:
                            v['colour'] = colors.to_hex(v['colour'])
                        except ValueError:
                            logger.warning(
                                f"Cannot convert colour {v['colour']} to hex, using default"
                            )
                            v['colour'] = random_colour()
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
        strati: bool = True,
        faults: bool = True,
        cmap: Optional[str] = None,
        model: Optional[GeologicalModel] = None,
        fault_colour: str = "black",
        paint_with: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        displacement_cmap=None,
        pyvista_kwargs: dict = {},
        scalar_bar: bool = False,
        name: Optional[str] = None,
    ):
        model = self._check_model(model)

        actors = []
        if strati:
            strati_surfaces = []
            surfaces = model.get_stratigraphic_surfaces()
            if cmap is None:
                cmap = self._build_stratigraphic_cmap(model)
            for s in surfaces:
                strati_surfaces.append(s.vtk())
            if name is None:
                object_name = 'model_surfaces'
            else:
                object_name = f'{name}_model_surfaces'
            object_name = self.increment_name(object_name)  # , 'model_surfaces')
            actors.append(
                self.add_mesh(
                    pv.MultiBlock(strati_surfaces).combine(),
                    cmap=cmap,
                    name=object_name,
                    **pyvista_kwargs,
                )
            )
            if not scalar_bar:
                self.remove_scalar_bar()
        if faults:
            fault_list = model.get_fault_surfaces()
            for f in fault_list:
                if name is None:
                    object_name = f'{f.name}_surface'
                if name is not None:
                    object_name = f'{name}_{f.name}_surface'
                object_name = self.increment_name(object_name)  # , 'fault_surfaces')
                actors.append(
                    self.add_mesh(f.vtk(), color=fault_colour, name=object_name, **pyvista_kwargs)
                )
        return actors

    def plot_vector_field(self, geological_feature, scale=1.0, name=None, pyvista_kwargs={}):
        if name is None:
            name = geological_feature.name + '_vector_field'
        name = self.increment_name(name)  # , 'vector_field')
        vectorfield = geological_feature.vector_field()
        return self.add_mesh(vectorfield.vtk(scale=scale), name=name, **pyvista_kwargs)

    def plot_data(
        self,
        feature,
        value=True,
        vector=True,
        scale=10,
        geom="arrow",
        name=None,
        pyvista_kwargs={},
    ):
        actors = []
        for d in feature.get_data():
            if isinstance(d, ValuePoints):
                if value:
                    if name is None:
                        object_name = d.name + '_values'
                    else:
                        object_name = f'{d.name}_values_{name}'
                    object_name = self.increment_name(object_name)  # , 'values')
                    actors.append(self.add_mesh(d.vtk(), name=object_name, **pyvista_kwargs))
            if isinstance(d, VectorPoints):
                if vector:
                    if name is None:
                        object_name = d.name + '_vectors'
                    else:
                        object_name = f'{d.name}_vectors_{name}'
                    object_name = self.increment_name(object_name)  # , 'vectors')
                    actors.append(
                        self.add_mesh(d.vtk(geom=geom, scale=scale), name=name, **pyvista_kwargs)
                    )

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
        name=None,
        pyvista_kwargs={},
    ):

        if surface:
            if name is None:
                surface_name = fault.name + '_surface'
            else:
                surface_name = f'{fault.name}_surface_{name}'
            surface_name = self.increment_name(surface_name)
            surface = fault.surfaces([0])[0]
            self.add_mesh(surface.vtk(), name=surface_name, **pyvista_kwargs)
        if slip_vector:
            if name is None:
                vector_name = fault.name + '_vector'
            else:
                vector_name = f'{fault.name}_vector_{name}'
            vector_name = self.increment_name(vector_name)

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
                name=vector_name,
                **pyvista_kwargs,
            )
        if fault_volume:
            if name is None:
                volume_name = fault.name + '_volume'
            else:
                volume_name = f'{fault.name}_volume_{name}'
            volume = fault.displacementfeature.scalar_field()
            volume.threshold(0.0)
            self.add_mesh(volume, name=volume_name, **pyvista_kwargs)

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
