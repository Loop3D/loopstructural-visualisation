# ruff: noqa: D102,D103,D107
"""PyVista Trame Viewer class for a Vue 3 client.

This class, derived from `pyvista.trame.ui.base_viewer`,
is intended for use with a trame application where the client type is "vue3".
Therefore, the `ui` method implemented by this class utilizes the API of Vuetify 3.
"""

from __future__ import annotations

from turtle import onclick
from typing import TYPE_CHECKING

from cycler import V
from trame.ui.vuetify3 import VAppLayout, SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify
from trame.widgets import trame
from pyvista.trame.views import PyVistaLocalView
from pyvista.trame.views import PyVistaRemoteLocalView
from pyvista.trame.views import PyVistaRemoteView

from pyvista.trame.ui.base_viewer import BaseViewer
from pyvista.trame.ui.vuetify3 import Viewer
from trame.widgets.vuetify3 import VTreeview


if TYPE_CHECKING:  # pragma: no cover
    from trame_client.ui.core import AbstractLayout

test = {}


class LoopViewer(Viewer):
    def __init__(self, *args, **kwargs):
        """Overwrite the pyvista trame layout to use a singlepage layout
        and add an object visibility menu to the drawer
        """
        super().__init__(*args, **kwargs)

    def make_layout(self, *args, **kwargs):

        return SinglePageWithDrawerLayout(*args, **kwargs)

    def ui(self, *args, **kwargs):
        with self.layout.content:
            return super().ui(*args, **kwargs)

    def toggle_visibility(self, **kwargs):
        """Toggle the visibility of an object in the plotter.
        this is a slot called by the state change, the kwargs are the current state
        so we need to check the keys and update accordingly
        """
        for k in kwargs.keys():
            object_name = k.split("__")[0]
            test[object_name] = kwargs[k]
            if object_name in self.plotter.objects:
                self.plotter.objects[object_name]["actor"].visibility = kwargs[k]
        self.update()
        # self.actors[k].visibility = not self.actors[k].visibility

    def object_menu(self):
        with self.layout.drawer as drawer:
            with vuetify.VCard():

                for k, a in self.plotter.objects.items():
                    drawer.server.state[f"{k}__visibility"] = True
                    drawer.server.state.change(f"{k}__visibility")(self.toggle_visibility)

                    with vuetify.VRow():
                        with vuetify.VCol():

                            vuetify.VCheckbox(
                                label=k,
                                v_model=(f"{k}__visibility"),
                                # click=(self.toggle_visibility("test")),
                            )
                # vuetify.VCheckbox(label="Test")
                # vuetify.VCheckbox(label="Test")
                # vuetify.VCheckbox(label="Test")
