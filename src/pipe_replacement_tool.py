import PySimpleGUI as sg
import os

from src.utils import is_valid_project_name, copy_shapefile
from src.tools import (
    process_shapefile,
    plot_metrics,
    save_edge_gdf_shapefile,
    spatial_autocorrelation_analysis,
    local_spatial_autocorrelation,
)

import ctypes
import platform

if platform.system() == "Windows":
    ctypes.windll.shcore.SetProcessDpiAwareness(1)


class PipeReplacementTool:
    def __init__(self):
        self.layout = [
            [
                sg.Menu(
                    [
                        [
                            "Projects",
                            ["New Project", "---", "Exit"],
                        ],
                    ]
                ),
                [
                    sg.Button("Step 1", disabled=True, key="-STEP1-"),
                    sg.Button("Step 2", disabled=True, key="-STEP2-"),
                    sg.Button("Step 3", disabled=True, key="-STEP3-"),
                    sg.Button("Step 4", disabled=True, key="-STEP4-"),
                ],
            ]
        ]

        self.step1_completed = False
        self.step2_completed = False

        self.window = sg.Window(
            "ΕΥΔΑΠ :: Pipe Replacement Tool",
            self.layout,
            resizable=False,
            finalize=True,
            icon="icon.ico",
        )

        self.projects_folder = os.path.join(os.path.expanduser("~"), "Pipe Replacement Tool Projects")

        self.project_name = None
        self.network_shapefile = None
        self.damage_shapefile = None

        self.fishnet_index = None
        self.select_square_size = None
        self.weight_avg_combined_metric = None
        self.weight_failures = None
        self.step2_output_path = None

        self.step1_result_shapefile = None

    def run(self):
        sg.set_options(icon="icon.ico")
        while True:
            event, values = self.window.read()
            print(event)
            if event == sg.WINDOW_CLOSED:
                break
            if event == "New Project":
                self.create_new_project()

            if self.network_shapefile and self.damage_shapefile:
                self.window["-STEP1-"].update(disabled=False)

            if event == "-STEP1-":
                self.step1()

            if event == "-STEP2-":
                self.step2()

            if event == "Exit":
                break

        self.window.close()

    def step1(self):
        layout = [
            [
                sg.Push(),
                sg.Text("Closeness Centrality Weight:"),
                sg.Slider(
                    range=(0, 1),
                    resolution=0.01,
                    orientation="h",
                    key="-Closeness Weight-",
                    enable_events=True,
                    default_value=0.33,
                ),
            ],
            [
                sg.Push(),
                sg.Text("Betweenness Centrality Weight:"),
                sg.Slider(
                    range=(0, 1),
                    resolution=0.01,
                    orientation="h",
                    key="-Betweenness Weight-",
                    enable_events=True,
                    default_value=0.33,
                    expand_x=True,
                ),
            ],
            [
                sg.Push(),
                sg.Text("Bridge Weight:"),
                sg.Slider(
                    range=(0, 1),
                    resolution=0.01,
                    orientation="h",
                    key="-Bridge Weight-",
                    enable_events=True,
                    default_value=0.34,
                ),
            ],
            [
                sg.Column(
                    [
                        [sg.Button("Calculate", key="-CALCULATE-")],
                    ],
                    justification="center",
                )
            ],
            [sg.Text("Calculating ...", key="-CALC Message-", visible=False)],
            [sg.Button("Proceed to Step 2", key="-PROCEED-", visible=False)],
        ]

        step1_window = sg.Window("Step 1", layout)

        while True:
            event, values = step1_window.read()
            if event == sg.WINDOW_CLOSED:
                break
            if event == "-PROCEED-":
                step1_window.close()
                break
            if event == "-CALCULATE-":
                closeness_weight = values["-Closeness Weight-"]
                betweenness_weight = values["-Betweenness Weight-"]
                bridge_weight = values["-Bridge Weight-"]
                total = closeness_weight + betweenness_weight + bridge_weight

                if not (0.99 <= total <= 1.01):
                    sg.popup_error("The sum of all sliders must be 1. Please adjust the sliders.")
                else:
                    step1_window["-CALC Message-"].update(visible=True)
                    step1_window.refresh()

                    gdf, G, nodes, edges, df_metrics = process_shapefile(
                        self.network_shapefile,
                        closeness_weight,
                        betweenness_weight,
                        bridge_weight,
                        os.path.join(self.projects_folder, self.project_name, ""),
                    )
                    plot_metrics(
                        gdf,
                        G,
                        nodes,
                        edges,
                        ["closeness", "betweenness", "bridge", "composite"],
                        8,
                        False,
                        os.path.join(self.projects_folder, self.project_name, ""),
                    )
                    output_path = os.path.join(
                        self.projects_folder,
                        self.project_name,
                        "shp_with_metrics",
                        "Pipes_WG_export_with_metrics.shp",
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_edge_gdf_shapefile(
                        edges,
                        output_path,
                    )
                    step1_window["-CALC Message-"].update(visible=False)
                    step1_window["-PROCEED-"].update(visible=True)
                    step1_window.refresh()
                    self.step1_completed = True
                    self.step1_result_shapefile = output_path
                    print(self.step1_result_shapefile)

                    # Update the main window to enable the next step
                    self.window["-STEP1-"].update(disabled=True)
                    self.window["-STEP2-"].update(disabled=False)

    def step2(self):
        layout = [
            [
                sg.Push(),
                sg.Text("Weighted Average combined metric:"),
                sg.Slider(
                    range=(0, 1),
                    resolution=0.01,
                    orientation="h",
                    key="-Weighted Average-",
                    enable_events=True,
                    default_value=0.5,
                ),
            ],
            [
                sg.Push(),
                sg.Text("Failures Weight:"),
                sg.Slider(
                    range=(0, 1),
                    resolution=0.01,
                    orientation="h",
                    key="-Failures Weight-",
                    enable_events=True,
                    default_value=0.5,
                ),
            ],
            [
                sg.Push(),
                sg.Text("Cell Lower Bound:"),
                sg.Slider(
                    range=(100, 1000),
                    resolution=100,
                    orientation="h",
                    key="-Cell Lower Bound-",
                    enable_events=True,
                    default_value=200,
                ),
            ],
            [
                sg.Push(),
                sg.Text("Cell Upper Bound:"),
                sg.Slider(
                    range=(200, 1000),
                    resolution=100,
                    orientation="h",
                    key="-Cell Upper Bound-",
                    enable_events=True,
                    default_value=1000,
                ),
            ],
            [
                sg.Column(
                    [
                        [sg.Button("Calculate", key="-CALCULATE-")],
                    ],
                    justification="center",
                )
            ],
            [sg.Text("Calculating ...", key="-CALC Message-", visible=False)],
            [sg.Text("", key="-Best Square Size-", visible=False)],
            [
                sg.Push(),
                sg.Text(
                    "Select custom square size",
                    key="-Custom Square Size Label-",
                    visible=False,
                ),
                sg.Slider(
                    range=(100, 1000),
                    resolution=100,
                    orientation="h",
                    key="-Custom Square Size-",
                    enable_events=True,
                    default_value=1000,
                    visible=False,
                ),
            ],
            [sg.Button("Continue Calculations", key="-CONTINUE-", visible=False)],
            [sg.Button("Proceed to Step 3", key="-PROCEED-", visible=False)],
        ]

        step2_window = sg.Window("Step 2", layout)

        while True:
            event, values = step2_window.read()
            if event == sg.WINDOW_CLOSED:
                break
            if event == "-PROCEED-":
                step2_window.close()
                break
            if event == "-CONTINUE-":
                self.select_square_size = values["-Custom Square Size-"]
                # Output --> σώζεται ένα shp file στο output path
                sorted_fishnet_df, results_pipe_clusters, fishnet_index = local_spatial_autocorrelation(
                    pipe_shapefile_path=self.step1_result_shapefile,
                    failures_shapefile_path=self.damage_shapefile,
                    weight_avg_combined_metric=self.weight_avg_combined_metric,
                    weight_failures=self.weight_failures,
                    select_square_size=self.select_square_size,
                    output_path=self.step2_output_path,
                )

                self.fishnet_index = fishnet_index

                print("Sorted fishnet df: ", sorted_fishnet_df)
                print("Results pipe clusters: ", results_pipe_clusters)
                step2_window["-CONTINUE-"].update(visible=False)
                step2_window["-PROCEED-"].update(visible=True)
                step2_window.refresh()
                self.step2_completed = True

                # Update the main window to enable the next step
                self.window["-STEP1-"].update(disabled=True)
                self.window["-STEP2-"].update(disabled=True)
                self.window["-STEP3-"].update(disabled=False)

            if event == "-CALCULATE-":
                weight_avg_combined_metric = values["-Weighted Average-"]
                failures_weight = values["-Failures Weight-"]
                cell_lower_bound = int(values["-Cell Lower Bound-"])
                cell_upper_bound = int(values["-Cell Upper Bound-"])
                total = weight_avg_combined_metric + failures_weight

                if total != 1:
                    sg.popup_error("The sum of all sliders must be 1. Please adjust the sliders.")
                else:
                    bounds = (cell_lower_bound, cell_upper_bound)
                    step2_window["-CALC Message-"].update(visible=True)
                    step2_window.refresh()

                    os.makedirs(
                        os.path.join(
                            self.projects_folder,
                            self.project_name,
                            "Fishnet_Grids",
                        ),
                        exist_ok=True,
                    )
                    output_path = os.path.join(
                        self.projects_folder,
                        self.project_name,
                        "Fishnet_Grids",
                        "",
                    )

                    results, best_square_size = spatial_autocorrelation_analysis(
                        pipe_shapefile_path=self.step1_result_shapefile,
                        failures_shapefile_path=self.damage_shapefile,
                        lower_bound_cell=cell_lower_bound,
                        upper_bound_cell=cell_upper_bound,
                        weight_avg_combined_metric=weight_avg_combined_metric,
                        weight_failures=failures_weight,
                        output_path=output_path,
                    )
                    print("Results: ", results)
                    print("Best square size: ", best_square_size)

                    step2_window["-CALC Message-"].update(visible=False)

                    # Passing the local variables to the class variables
                    # to use them in the next steps
                    self.weight_avg_combined_metric = weight_avg_combined_metric
                    self.weight_failures = failures_weight
                    self.step2_output_path = output_path

                    # Ask the user if they want to keep the best square size or input a custom one
                    step2_window["-CALCULATE-"].update(visible=False)
                    step2_window["-Best Square Size-"].update(f"Best square size: {best_square_size}", visible=True)
                    step2_window["-Custom Square Size Label-"].update(visible=True)
                    step2_window["-Custom Square Size-"].update(best_square_size, range=bounds, visible=True)
                    step2_window["-CONTINUE-"].update(visible=True)

                    ### (εδώ να βγαίνει μήνυμα που να τον ρωτάει αν θα κρατάει αυτό ή να βάλει δικό του)
                    self.select_square_size = best_square_size

    def create_new_project(self):
        layout = [
            [sg.Text("New Project Name:")],
            [sg.Input()],
            [sg.Text("Select network shapefile")],
            [sg.Input(), sg.FileBrowse(file_types=(("Shapefiles", "*.shp"),))],
            [sg.Text("Select damage shapefile")],
            [sg.Input(), sg.FileBrowse(file_types=(("Shapefiles", "*.shp"),))],
            [sg.Column([[sg.Button("Create Project")]], justification="center")],
        ]

        new_project_window = sg.Window("Create Project", layout, finalize=True)

        while True:
            event, values = new_project_window.read()
            if event == sg.WINDOW_CLOSED:
                break
            if event == "Create Project":
                project_name = values[0]
                self.project_name = project_name
                network_shapefile_path = values[1]
                damage_shapefile_path = values[2]

                if is_valid_project_name(project_name):
                    print("Creating new project:", project_name)
                    new_project_window.close()

                    project_folder = os.path.join(self.projects_folder, project_name)
                    os.makedirs(project_folder, exist_ok=True)

                    self.network_shapefile = copy_shapefile("network", network_shapefile_path, project_folder)
                    self.damage_shapefile = copy_shapefile("damage", damage_shapefile_path, project_folder)

                else:
                    sg.popup("Invalid project name. Please try again.")
